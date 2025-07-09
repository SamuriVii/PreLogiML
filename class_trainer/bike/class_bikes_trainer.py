from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(180)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_dto import BikesData, ModelStatus, CEST
from datetime import datetime, timedelta
from shared.db_conn import SessionLocal
from sqlalchemy import select, inspect
from shared.db_utils import save_log
from sqlalchemy.orm import Session

# --- Ustawienia ---
MODEL_DIR = '/app/shared/classification/models/'
MODEL_SAVE_PATH_BINARY = os.path.join(MODEL_DIR, 'bike_binary_model_new.pkl')
MODEL_SAVE_PATH_MULTICLASS = os.path.join(MODEL_DIR, 'bike_multiclass_model_new.pkl')
MODEL_SAVE_PATH_REGRESSION = os.path.join(MODEL_DIR, 'bike_regression_model_new.pkl')
TRAINING_DATA_INTERVAL_DAYS = 60 # Okres danych do treningu (np. 60 dni)
TRAINING_CYCLE_INTERVAL_HOURS = 12 # Jak często ma się odbywać trening (np. co 12 godzin)

# Upewnij się, że katalog na modele istnieje
os.makedirs(MODEL_DIR, exist_ok=True)

# +-------------------------------------------------+
# |           KOLUMNY DO TRENINGU MODELI            |
# |         Trenowanie Modeli Klasyfikacji          |
# +-------------------------------------------------+

# Globalne listy kolumn, aby były dostępne w obu funkcjach
FEATURE_COLUMNS_BASE = [
    'capacity', 'manual_bikes_available', 'electric_bikes_available',
    'temperature', 'feelslike', 'humidity', 'wind_kph', 'precip_mm',
    'cloud', 'visibility_km', 'uv_index', 'daylight', 'weather_condition',
    'fine_particles_pm2_5', 'coarse_particles_pm10', 'carbon_monoxide_ppb',
    'nitrogen_dioxide_ppb', 'ozone_ppb', 'sulfur_dioxide_ppb'
]

TARGET_CALCULATION_COLUMNS = ['bikes_available', 'capacity'] 

# +-------------------------------------------------+
# |         FUNKCJE POMOCNICZE DO TRENINGU          |
# |         Trenowanie Modeli Klasyfikacji          |
# +-------------------------------------------------+

# Pobiera dane rowerowe z bazy danych dla określonego interwału czasowego, wybierając tylko niezbędne kolumny do treningu i obliczenia celów.
def fetch_bike_data_for_training(session: Session, interval_days: int) -> pd.DataFrame:

    start_time = datetime.now(CEST) - timedelta(days=interval_days)

    # Dynamiczne budowanie listy kolumn do pobrania z DTO
    mapper = inspect(BikesData)
    columns_to_select_from_dto = []
    
    # Połącz listy cech i kolumn do obliczenia celów i usuń duplikaty
    all_needed_columns = list(set(FEATURE_COLUMNS_BASE + TARGET_CALCULATION_COLUMNS))

    for col_name in all_needed_columns:
        if hasattr(BikesData, col_name): # Sprawdź, czy DTO posiada taką kolumnę
            columns_to_select_from_dto.append(getattr(BikesData, col_name))
        else:
            save_log("class_bike_trainer", "warning", f"Kolumna '{col_name}' zdefiniowana jako potrzebna, ale nie istnieje w BikesData DTO. Pomijam.")

    if not columns_to_select_from_dto:
        save_log("class_bike_trainer", "error", "Brak kolumn do pobrania z BikesData DTO. Sprawdź definicje kolumn i DTO.")
        return pd.DataFrame() # Zwróć pusty DataFrame

    query_result = session.execute(
        select(*columns_to_select_from_dto) # Użyj gwiazdki, aby rozpakować listę kolumn
        .filter(BikesData.timestamp >= start_time)
    ).all() # Pobieramy wyniki jako listę krotek

    # Konwertujemy listę krotek na DataFrame, używając nazw kolumn
    df = pd.DataFrame(query_result, columns=[c.name for c in columns_to_select_from_dto])

    save_log("class_bike_trainer", "info", f"Pobrano {len(df)} rekordów danych rowerowych do treningu z ostatnich {interval_days} dni.")
    return df

# Przygotowuje cechy i zmienne docelowe do treningu modeli dla danych rowerowych. Zwraca przetworzony DataFrame z cechami, zmienne docelowe oraz LabelEncodery.
def prepare_features_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder, LabelEncoder]:

    df_processed = df.copy()

    # Sprawdzenie obecności kluczowych kolumn docelowych
    if not all(col in df_processed.columns for col in TARGET_CALCULATION_COLUMNS):
        missing = [col for col in TARGET_CALCULATION_COLUMNS if col not in df_processed.columns]
        save_log("class_bike_trainer", "error", f"Brak kluczowych kolumn docelowych {missing} w danych. Nie można przygotować celów.")
        raise ValueError(f"Brak wymaganych kolumn dla zmiennych docelowych: {missing}.")

    # --- DODATKOWE LOGOWANIE DLA DIAGNOSTYKI ---
    save_log("class_bike_trainer", "info", f"prepare_features_and_targets: Kształt df_processed po pobraniu: {df_processed.shape}")
    save_log("class_bike_trainer", "info", f"prepare_features_and_targets: Kolumny df_processed: {df_processed.columns.tolist()}")

    # Obsługa brakujących wartości
    initial_rows = len(df_processed)
    
    current_columns = df_processed.columns.tolist() 
    for col in current_columns:
        if col not in df_processed.columns: # Sprawdź, czy kolumna nadal istnieje po ewentualnym dropna wcześniej
            continue

        save_log("class_bike_trainer", "debug", f"prepare_features_and_targets: Przetwarzam kolumnę '{col}' o typie: {df_processed[col].dtype}")

        if df_processed[col].dtype in ['int64', 'float64']:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                if pd.isna(mean_val):
                    df_processed[col] = df_processed[col].fillna(0)
                    save_log("class_bike_trainer", "warning", f"Kolumna '{col}' zawierała tylko NaN, wypełniono 0.")
                else:
                    df_processed[col] = df_processed[col].fillna(mean_val)
        elif df_processed[col].dtype == 'object':
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                if not mode_val.empty:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                else:
                    df_processed[col] = df_processed[col].fillna('unknown') 
                    save_log("class_bike_trainer", "warning", f"Kolumna '{col}' stringowa zawierała tylko NaN, wypełniono 'unknown'.")

    # Dodatkowe usunięcie NaN, jeśli po wypełnieniu nadal istnieją (np. kolumny tylko z NaN)
    df_processed.dropna(inplace=True)

    if len(df_processed) < initial_rows:
        save_log("class_bike_trainer", "warning", f"Wypełniono/Usunięto {initial_rows - len(df_processed)} wierszy z brakującymi danymi podczas przygotowania cech.")

    if df_processed.empty:
        save_log("class_bike_trainer", "error", "Brak danych po usunięciu wierszy z brakującymi wartościami. Nie można trenować modelu.")
        raise ValueError("Brak danych po usunięciu wierszy z brakującymi wartościami.")

    # Enkodowanie 'daylight'
    daylight_le = None
    if 'daylight' in df_processed.columns:
        daylight_le = LabelEncoder()
        known_daylight_labels = ['yes', 'no']
        df_processed['daylight'] = df_processed['daylight'].apply(lambda x: x if x in known_daylight_labels else 'no')
        df_processed['daylight_encoded'] = daylight_le.fit_transform(df_processed['daylight'])
        df_processed = df_processed.drop('daylight', axis=1)
    
    # Enkodowanie 'weather_condition'
    weather_le = None
    if 'weather_condition' in df_processed.columns:
        weather_le = LabelEncoder()
        df_processed['weather_condition_encoded'] = weather_le.fit_transform(df_processed['weather_condition'])
        df_processed = df_processed.drop('weather_condition', axis=1)
    
    # Przygotowanie zmiennych docelowych
    df_processed['bikes_available_reg'] = df_processed['bikes_available'] # Regresja - dokładna liczba
    
    # Klasyfikacja binarna: low_bikes (0 = wystarczająco, 1 = mało/brak)
    # Upewnij się, że 'capacity' jest > 0, aby uniknąć ZeroDivisionError
    df_processed['low_bikes'] = df_processed.apply(
        lambda row: 1 if row['capacity'] > 0 and row['bikes_available'] < row['capacity'] * 0.25 else 0, axis=1
    ).astype(int)
    
    # Klasyfikacja wieloklasowa: bike_level (0-brak, 1-mało, 2-średnio, 3-dużo)
    def categorize_bike_level(row):
        bikes = row['bikes_available']
        capacity = row['capacity']
        ratio = bikes / capacity if capacity > 0 else 0
        
        if ratio == 0:
            return 0 # brak
        elif ratio <= 0.33:
            return 1 # mało
        elif ratio <= 0.66:
            return 2 # średnio
        else:
            return 3 # dużo
    
    df_processed['bike_level'] = df_processed.apply(categorize_bike_level, axis=1)
    
    # Aktualizacja listy kolumn cech dla finalnego DataFrame X
    final_feature_columns = [f for f in FEATURE_COLUMNS_BASE if f in df_processed.columns]
    
    # Usuwamy oryginalne kolumny kategoryczne, jeśli zostały zakodowane i dodajemy ich zakodowane odpowiedniki.
    if 'daylight' in df_processed.columns and 'daylight_encoded' in df_processed.columns:
        if 'daylight' in final_feature_columns:
            final_feature_columns.remove('daylight')
        final_feature_columns.append('daylight_encoded')

    if 'weather_condition' in df_processed.columns and 'weather_condition_encoded' in df_processed.columns:
        if 'weather_condition' in final_feature_columns:
            final_feature_columns.remove('weather_condition')
        final_feature_columns.append('weather_condition_encoded')

    # Przygotowanie cech (X) - upewnij się, że używasz zaktualizowanej listy
    X = df_processed[[col for col in final_feature_columns if col in df_processed.columns]].copy()
    
    # Zmienne docelowe
    y_binary = df_processed['low_bikes']
    y_multiclass = df_processed['bike_level']
    y_regression = df_processed['bikes_available_reg']
    
    save_log("class_bike_trainer", "info", f"Pomyślnie przygotowano cechy i cele do treningu. Liczba próbek: {len(X)}.")
    return X, y_binary, y_multiclass, y_regression, daylight_le, weather_le

# Trenuje i zapisuje model klasyfikacji/regresji. Zwraca metrykę jakości (Accuracy dla klasyfikacji, RMSE dla regresji).
def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_type: str, model_path: str, label_encoders: dict = None) -> float:
    
    # Sprawdzenie minimalnej liczby próbek dla podziału danych
    if len(X) < 2 or len(y) < 2: # Minimalnie 2 próbki do train_test_split
        save_log("class_bike_trainer", "warning", f"Zbyt mało danych do treningu dla typu modelu: {model_type}. Wymagane min. 2 próbki, dostępne {len(X)}. Pomijam trening.")
        return 0.0 # Zwróć 0 jako metrykę jakości

    # Upewnij się, że y ma co najmniej dwie unikalne klasy dla klasyfikacji
    if model_type in ['binary_classification', 'multiclass_classification'] and len(y.unique()) < 2:
        save_log("class_bike_trainer", "warning", f"Zbyt mało unikalnych klas ({len(y.unique())}) dla klasyfikacji {model_type}. Wymagane min. 2. Pomijam trening.")
        return 0.0

    save_log("class_bike_trainer", "info", f"Rozpoczynam trening modelu {model_type}...")

    # Podział danych
    # Użyj stratify dla klasyfikacji, jeśli są co najmniej 2 unikalne klasy
    if model_type in ['binary_classification', 'multiclass_classification'] and len(y.unique()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizacja danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_model = None
    best_score = -float('inf') if model_type != 'regression' else float('inf') # Accuracy vs RMSE
    best_model_name = ""
    
    if model_type == 'binary_classification':
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, solver='liblinear') # Użyj solvera, który wspiera binary
        }
        scoring_metric = 'accuracy'
        label_mapping = {0: "standardowo", 1: "mało"} # Dla zapisu w modelu
    elif model_type == 'multiclass_classification':
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, multi_class='ovr') # 'ovr' lub 'multinomial'
        }
        scoring_metric = 'accuracy'
        label_mapping = {0: "brak", 1: "mała dostępność", 2: "standardowa dostępność", 3: "wysoka dostępność"} # Dla zapisu w modelu
    elif model_type == 'regression':
        models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        scoring_metric = 'neg_mean_squared_error'
        label_mapping = None # Dla regresji nie ma label_mapping
    else:
        save_log("class_bike_trainer", "error", f"Nieznany typ modelu: {model_type}")
        return 0.0

    for name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=scoring_metric)
            
            # Trenowanie na pełnych danych treningowych
            model.fit(X_train_scaled, y_train)
            
            # Ocena na zbiorze testowym
            y_pred = model.predict(X_test_scaled)

            if model_type in ['binary_classification', 'multiclass_classification']:
                test_score = accuracy_score(y_test, y_pred)
                save_log("class_bike_trainer", "info", f"{name} - CV Accuracy: {cv_scores.mean():.4f}, Test Accuracy: {test_score:.4f}")
                if test_score > best_score:
                    best_score = test_score
                    best_model = model
                    best_model_name = name
            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                save_log("class_bike_trainer", "info", f"{name} - CV RMSE: {np.sqrt(-cv_scores.mean()):.4f}, Test RMSE: {rmse:.4f}, Test R²: {r2:.4f}")
                if rmse < best_score: # Dla RMSE szukamy minimum
                    best_score = rmse
                    best_model = model
                    best_model_name = name
        except Exception as e:
            save_log("class_bike_trainer", "error", f"Błąd podczas treningu modelu {name} ({model_type}): {e}")
            continue

    if best_model is None:
        save_log("class_bike_trainer", "error", f"Nie udało się wytrenować żadnego modelu dla {model_type}.")
        return 0.0

    # Zapisanie najlepszego modelu
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'model_type': model_type,
        'model_name': best_model_name,
        'label_encoders': label_encoders, # Zapisz LabelEncoders
        'quality_metric': best_score,
        'label_mapping': label_mapping # Zapisz mapowanie tekstowe dla klasyfikacji
    }
    
    # Dodatkowe metryki do zapisu w zależności od typu modelu
    if model_type in ['binary_classification', 'multiclass_classification']:
        model_data['test_accuracy'] = best_score
    elif model_type == 'regression':
        model_data['test_rmse'] = best_score
        model_data['test_r2'] = r2_score(y_test, best_model.predict(X_test_scaled))

    joblib.dump(model_data, model_path)
    save_log("class_bike_trainer", "info", f"Model '{best_model_name}' ({model_type}) zapisany do: {model_path} z metryką jakości: {best_score:.4f}.")
    
    return best_score

# Aktualizuje status modelu w bazie danych.
def update_model_status(session: Session, model_name: str, quality_metric: float, metric_type: str):
    try:
        model_status_entry = session.execute(
            select(ModelStatus).filter_by(model_name=model_name)
        ).scalar_one_or_none()

        current_version = 0
        if model_status_entry:
            current_version = model_status_entry.version if model_status_entry.version is not None else 0
            model_status_entry.is_new_model_available = True
            model_status_entry.last_updated = datetime.now(CEST)
            model_status_entry.quality_metric = float(quality_metric)
            model_status_entry.version = current_version + 1
            # Upewnij się, że pole 'metric_type' istnieje w Twoim ModelStatus DTO
            if hasattr(model_status_entry, 'metric_type'):
                model_status_entry.metric_type = metric_type 
            save_log(f"trainer_{model_name.replace('_', '-')}", "info", 
                     f"Zaktualizowano status modelu '{model_name}' w bazie danych. "
                     f"Nowa wersja: {model_status_entry.version}, Metryka: {metric_type}={quality_metric:.4f}.")
        else:
            new_entry = ModelStatus(
                model_name=model_name,
                is_new_model_available=True,
                last_updated=datetime.now(CEST),
                quality_metric=float(quality_metric),
                version=1,
            )
            session.add(new_entry)
            save_log(f"trainer_{model_name.replace('_', '-')}", "warning", 
                     f"Utworzono nowy wpis statusu dla modelu '{model_name}'. "
                     f"Wersja: 1, Metryka: {metric_type}={quality_metric:.4f}.")
        
        session.commit()
    except Exception as e:
        session.rollback()
        save_log(f"trainer_{model_name.replace('_', '-')}", "error", 
                     f"Błąd podczas aktualizacji statusu modelu '{model_name}' w bazie danych: {e}")

# +-------------------------------------+
# |     GŁÓWNA FUNKCJA TRENINGU ML      |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

# Główna funkcja do uruchamiania cyklu treningowego dla klasyfikatorów rowerowych.
def run_bike_classification_training_cycle():
    save_log("class_bike_trainer", "info", "🚀 Rozpoczynam cykl treningowy dla klasyfikatorów rowerowych...")
    
    db_session = SessionLocal()
    try:
        # 1. Pobierz dane do treningu
        save_log("class_bike_trainer", "info", f"Pobieranie danych rowerowych z bazy za ostatnie {TRAINING_DATA_INTERVAL_DAYS} dni...")
        enriched_df = fetch_bike_data_for_training(db_session, TRAINING_DATA_INTERVAL_DAYS)
        
        if enriched_df.empty:
            save_log("class_bike_trainer", "warning", "Brak danych rowerowych do treningu w określonym interwale. Pomijam trening.")
            return

        # 2. Przygotuj cechy i cele
        save_log("class_bike_trainer", "info", "Przygotowywanie cech i celów dla rowerów...")
        try:
            X, y_binary, y_multiclass, y_regression, daylight_le, weather_le = prepare_features_and_targets(enriched_df)
        except ValueError as e:
            save_log("class_bike_trainer", "error", f"Błąd podczas przygotowania danych: {e}. Pomijam trening.")
            return

        if X.empty:
            save_log("class_bike_trainer", "warning", "Brak danych po przygotowaniu cech. Pomijam trening.")
            return

        label_encoders = {
            'daylight_le': daylight_le,
            'daylight_categories': daylight_le.classes_.tolist() if daylight_le else [], # Dodaj kategorie dla lepszej diagnostyki
            'weather_le': weather_le,
            'weather_categories': weather_le.classes_.tolist() if weather_le else [] # Dodaj kategorie
        }

        # 3. Trenowanie i zapisywanie modeli
        
        # Model binarny
        binary_score = train_and_save_model(X, y_binary, 'binary_classification', MODEL_SAVE_PATH_BINARY, label_encoders)
        update_model_status(db_session, 'bikes_binary_classifier', binary_score, 'accuracy')

        # Model wieloklasowy
        multiclass_score = train_and_save_model(X, y_multiclass, 'multiclass_classification', MODEL_SAVE_PATH_MULTICLASS, label_encoders)
        update_model_status(db_session, 'bikes_multiclass_classifier', multiclass_score, 'accuracy')
        
        # Model regresji
        regression_score = train_and_save_model(X, y_regression, 'regression', MODEL_SAVE_PATH_REGRESSION, label_encoders)
        update_model_status(db_session, 'bikes_regression_predictor', regression_score, 'rmse')
        
        save_log("class_bike_trainer", "info", "✅ Cykl treningowy dla klasyfikatorów rowerowych zakończony pomyślnie.")

    except Exception as e:
        save_log("class_bike_trainer", "error", f"❌ Krytyczny błąd w cyklu treningowym klasyfikatorów rowerowych: {e}")
    finally:
        db_session.close()

# +-------------------------------------+
# |     GŁÓWNA FUNKCJA WYKONUJĄCA       |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

if __name__ == "__main__":
    while True:
        run_bike_classification_training_cycle()
        save_log("class_bike_trainer", "info", f"Kolejny cykl treningowy za {TRAINING_CYCLE_INTERVAL_HOURS} godzin.")
        time.sleep(TRAINING_CYCLE_INTERVAL_HOURS * 3600)
