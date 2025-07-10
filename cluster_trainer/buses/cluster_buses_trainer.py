from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import joblib
import os

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_dto import BusesData, ModelStatus, CEST
from shared.db_conn import SessionLocal
from shared.db_utils import save_log
from sqlalchemy.orm import Session
from sqlalchemy import select

# --- Ustawienia ---
MODEL_SAVE_PATH_BUSES = '/app/shared/clusterization/models/buses_kmeans_new.pkl'
MAX_CLUSTERS = 8
TRAINING_DATA_INTERVAL_DAYS = 30

# Upewnij się, że katalog na modele istnieje
os.makedirs(os.path.dirname(MODEL_SAVE_PATH_BUSES), exist_ok=True)

# +-------------------------------------------------+
# |         FUNKCJE POMOCNICZE DO TRENINGU          |
# |         Trenowanie Modeli Klasteryzacji         |
# +-------------------------------------------------+

# Pobiera dane autobusowe z bazy danych dla określonego interwału czasowego, wybierając tylko kolumny potrzebne do klasteryzacji.
def fetch_bus_data_for_training(session: Session, interval_days: int) -> pd.DataFrame:
    start_time = datetime.now(CEST) - timedelta(days=interval_days)

    columns_to_fetch = [
        BusesData.average_delay_seconds, BusesData.maximum_delay_seconds, BusesData.minimum_delay_seconds,
        BusesData.delay_standard_deviation, BusesData.delay_range_seconds, BusesData.on_time_stop_ratio, 
        BusesData.delay_consistency_score, BusesData.stops_count,
        BusesData.temperature, BusesData.wind_kph, BusesData.precip_mm, BusesData.humidity,
        BusesData.weather_condition,
        BusesData.fine_particles_pm2_5, BusesData.coarse_particles_pm10
    ]

    # Wykonujemy zapytanie, aby pobrać tylko te kolumny
    query_result = session.execute(
        select(*columns_to_fetch)
        .filter(BusesData.timestamp >= start_time)
    ).fetchall()
    
    # Tworzymy DataFrame z pobranych danych
    df = pd.DataFrame(query_result, columns=[col.name for col in columns_to_fetch])

    save_log("cluster_buses_trainer", "info", f"Pobrano {len(df)} rekordów danych autobusowych do treningu z ostatnich {interval_days} dni.")
    return df

# Przygotowuje cechy do klasteryzacji dla danych autobusowych z DataFrame. Zwraca przetworzony DataFrame i LabelEncoder.
def prepare_bus_features_for_training(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:

    # Kolumny cech, które będą użyte do treningu
    features = [
        'average_delay_seconds', 'maximum_delay_seconds', 'minimum_delay_seconds',
        'delay_standard_deviation', 'delay_range_seconds', 'on_time_stop_ratio', 
        'delay_consistency_score', 'stops_count',
        'temperature', 'wind_kph', 'precip_mm', 'humidity',
        'weather_condition',
        'fine_particles_pm2_5', 'coarse_particles_pm10'
    ]
    
    # Upewnij się, że wszystkie wymagane kolumny istnieją w pobranym DataFrame
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        save_log("cluster_buses_trainer", "error", f"Brakuje kolumn w danych wejściowych do treningu: {missing_cols}")
        raise ValueError(f"Brakuje kolumn w danych wejściowych do treningu: {missing_cols}")

    df_processed = df[features].copy()
    
    # Obsługa brakujących wartości (dropna)
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    if len(df_processed) < initial_rows:
        save_log("cluster_buses_trainer", "warning", f"Usunięto {initial_rows - len(df_processed)} wierszy z brakującymi danymi podczas przygotowania cech.")

    if df_processed.empty:
        save_log("cluster_buses_trainer", "error", "Brak danych po usunięciu wierszy z brakującymi wartościami. Nie można trenować modelu.")
        raise ValueError("Brak danych po usunięciu wierszy z brakującymi wartościami.")

    # Enkodowanie weather_condition
    le = LabelEncoder()
    df_processed['weather_condition_encoded'] = le.fit_transform(df_processed['weather_condition'])
    df_processed = df_processed.drop('weather_condition', axis=1)
    
    save_log("cluster_buses_trainer", "info", f"Pomyślnie przygotowano cechy do treningu. Liczba próbek: {len(df_processed)}.")
    return df_processed, le

# Znajduje optymalną liczbę klastrów używając silhouette score.
def find_optimal_clusters_and_score(X: pd.DataFrame, max_clusters: int = 8) -> tuple[int, float]:

    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    if len(X) < 2: 
        save_log("cluster_buses_trainer", "warning", "Zbyt mało próbek do obliczenia Silhouette Score. Zwracam domyślne k=2 i score 0.0.")
        return 2, 0.0 
    
    for k in K_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            # Silhouette score wymaga co najmniej 2 klastrów i więcej niż 1 próbki na klaster
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1.0) # Oznacz jako niepoprawny wynik
        except Exception as e:
            save_log("cluster_buses_trainer", "error", f"Błąd podczas obliczania Silhouette Score dla k={k}: {e}")
            silhouette_scores.append(-1.0)

    if not silhouette_scores or all(s == -1.0 for s in silhouette_scores):
        save_log("cluster_buses_trainer", "warning", "Nie można znaleźć optymalnej liczby klastrów. Wszystkie Silhouette Scores są niepoprawne. Zwracam domyślne k=2 i score 0.0.")
        return 2, 0.0

    best_k_index = np.argmax(silhouette_scores)
    best_k = K_range[best_k_index]
    best_silhouette_score = silhouette_scores[best_k_index]
    
    save_log("cluster_buses_trainer", "info", f"Znaleziono optymalne k={best_k} z Silhouette Score: {best_silhouette_score:.3f}.")
    return best_k, best_silhouette_score

# Trenuje model KMeans dla autobusów i zapisuje go do pliku.
def train_and_save_bus_kmeans_model(X: pd.DataFrame, n_clusters: int, scaler: StandardScaler, label_encoder: LabelEncoder, model_path: str) -> float:
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Oblicz silhouette score dla zapisanego modelu
    final_silhouette_score = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0.0

    model_data = {
        'kmeans': kmeans,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist(),
        'n_clusters': n_clusters,
        'silhouette_score': final_silhouette_score
    }
    
    joblib.dump(model_data, model_path)
    save_log("cluster_buses_trainer", "info", f"Model autobusowy zapisany do: {model_path} z Silhouette Score: {final_silhouette_score:.3f}.")
    return final_silhouette_score

# Aktualizuje status modelu w bazie danych.
def update_model_status_in_db(session: Session, model_name: str, quality_metric: float, version: int):
    try:
        # Konwersja quality_metric na standardowy Python float
        quality_metric_standard_float = float(quality_metric) 

        model_status_entry = session.execute(
            select(ModelStatus).filter_by(model_name=model_name)
        ).scalar_one_or_none()

        if model_status_entry:
            model_status_entry.is_new_model_available = True
            model_status_entry.last_updated = datetime.now(CEST)
            model_status_entry.quality_metric = quality_metric_standard_float
            model_status_entry.version = version + 1
            save_log("cluster_buses_trainer", "info", f"Zaktualizowano status modelu '{model_name}' w bazie danych. Nowa wersja: {model_status_entry.version}.")
        else:
            new_entry = ModelStatus(
                model_name=model_name,
                is_new_model_available=True,
                last_updated=datetime.now(CEST),
                quality_metric=quality_metric_standard_float,
                version=1
            )
            session.add(new_entry)
            save_log("cluster_buses_trainer", "warning", f"Utworzono nowy wpis statusu dla modelu '{model_name}'.")

        session.commit()
    except Exception as e:
        session.rollback()
        save_log("cluster_buses_trainer", "error", f"Błąd podczas aktualizacji statusu modelu '{model_name}' w bazie danych: {e}")


# +-------------------------------------------------+
# |           GŁÓWNA FUNKCJA DLA PROCESU            |
# |         Trenowanie Modeli Klasteryzacji         |
# +-------------------------------------------------+

# Główna funkcja do uruchamiania cyklu treningowego dla klastrów autobusowych.
def run_bus_cluster_training_cycle():
    save_log("cluster_buses_trainer", "info", "🚀 Rozpoczynam cykl treningowy dla klastrów autobusowych...")
    
    db_session = SessionLocal()
    try:
        # 1. Pobierz dane do treningu
        save_log("cluster_buses_trainer", "info", f"Pobieranie danych autobusowych z bazy za ostatnie {TRAINING_DATA_INTERVAL_DAYS} dni...")
        enriched_df = fetch_bus_data_for_training(db_session, TRAINING_DATA_INTERVAL_DAYS)
        
        if enriched_df.empty:
            save_log("cluster_buses_trainer", "warning", "Brak danych autobusowych do treningu w określonym interwale. Pomijam trening.")
            return

        # 2. Przygotuj cechy
        save_log("cluster_buses_trainer", "info", "Przygotowywanie cech autobusowych...")
        df_features, label_encoder = prepare_bus_features_for_training(enriched_df)
        
        # 3. Normalizacja danych
        save_log("cluster_buses_trainer", "info", "Normalizacja danych autobusowych...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)
        X_scaled_df = pd.DataFrame(X_scaled, columns=df_features.columns)

        # 4. Znajdź optymalną liczbę klastrów
        save_log("cluster_buses_trainer", "info", "Szukanie optymalnej liczby klastrów dla autobusów...")
        optimal_k, silhouette_score_val = find_optimal_clusters_and_score(X_scaled_df, MAX_CLUSTERS)
        
        # 5. Trenuj i zapisz model
        save_log("cluster_buses_trainer", "info", f"Trenowanie i zapisywanie modelu autobusowego z {optimal_k} klastrami...")
        final_silhouette_score = train_and_save_bus_kmeans_model(
            X_scaled_df, optimal_k, scaler, label_encoder, MODEL_SAVE_PATH_BUSES
        )
        
        # 6. Zaktualizuj status modelu w bazie danych
        current_version_entry = db_session.execute(
            select(ModelStatus.version).filter_by(model_name='buses_kmeans')
        ).scalar_one_or_none()
        current_version = current_version_entry if current_version_entry is not None else 0

        update_model_status_in_db(db_session, 'buses_kmeans', final_silhouette_score, current_version)
        
        save_log("cluster_buses_trainer", "info", "✅ Cykl treningowy dla klastrów autobusowych zakończony pomyślnie.")

    except Exception as e:
        save_log("cluster_buses_trainer", "error", f"❌ Krytyczny błąd w cyklu treningowym klastrów autobusowych: {e}")
    finally:
        db_session.close()

# +-------------------------------------+
# |     GŁÓWNA FUNKCJA WYKONUJĄCA       |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

if __name__ == "__main__":
    run_bus_cluster_training_cycle()
    save_log("cluster_buses_trainer", "info", f"Kolejny cykl treningowy za {24} godzin.")
