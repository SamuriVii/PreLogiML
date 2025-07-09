from shared.db_utils import save_log
from typing import Dict, Optional
import pandas as pd
import joblib
import os

# +--------------------------------------------------+
# |      KLASA MODELU ROWEROWEGO KLASTERYZACJI       |
# |            Funkcje do zarządzania                |
# +--------------------------------------------------+

"""
Klasa odpowiedzialna za predykcję klastra dla stacji rowerowych.
Wykorzystuje wytrenowany model K-Means do grupowania stacji
o podobnych charakterystykach operacyjnych i warunkach środowiskowych.

Klasteryzowane cechy:
- Dane o stacji: 'bikes_available', 'docks_available', 'capacity',
    'manual_bikes_available', 'electric_bikes_available'.
    Określają bieżący stan i pojemność stacji, kluczowe dla zarządzania flotą.
- Dane pogodowe: 'temperature', 'wind_kph', 'precip_mm', 'humidity',
    'weather_condition'.
    Pogoda znacząco wpływa na użytkowanie rowerów, pozwalając na grupowanie
    stacji według ich reakcji na różne warunki atmosferyczne.
- Dane o jakości powietrza: 'fine_particles_pm2_5', 'coarse_particles_pm10'.
    Zanieczyszczenie powietrza może wpływać na decyzje użytkowników,
    zmieniając wzorce korzystania ze stacji.

Celem jest identyfikacja typowych zachowań stacji w różnych warunkach,
co wspiera optymalizację dystrybucji rowerów i doków.
"""

class BikeStationClusterPredictor:
    # Inicjalizuje predyktor z wczytanym modelem
    def __init__(self, model_path='/app/shared/clusterization/models/bikes_kmeans.pkl'):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace('.pkl', '') 
        self.new_model_path = model_path.replace('.pkl', '_new.pkl')
        self.model_data = None
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        # Przy inicjalizacji, jeśli istnieje "_new" plik, zastąp nim główny
        if os.path.exists(self.new_model_path):
            save_log(self.model_name, "info", f"Znaleziono nowy model {self.new_model_path} przy starcie. Przenoszę go na główną ścieżkę.")
            try:
                if os.path.exists(self.model_path):
                    os.remove(self.model_path) # Usuń stary model, jeśli istnieje
                os.rename(self.new_model_path, self.model_path) # Przemianuj nowy na główny
            except Exception as e:
                save_log(self.model_name, "error", f"Błąd przy przenoszeniu {self.new_model_path} na {self.model_path} podczas startu: {e}")
        
        self.load_model()
    
    # Wczytuje zapisany model i wszystkie komponenty
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model nie istnieje: {self.model_path}")
                return False
                
            print(f"🔄 Ładowanie modelu z: {self.model_path}")
            self.model_data = joblib.load(self.model_path)
            
            self.kmeans = self.model_data['kmeans']
            self.scaler = self.model_data['scaler']
            self.label_encoder = self.model_data['label_encoder']
            self.feature_names = self.model_data['feature_names']
            
            print(f"✅ Model załadowany pomyślnie!")
            print(f"   Liczba klastrów: {self.model_data['n_clusters']}")
            print(f"   Silhouette Score: {self.model_data['silhouette_score']:.3f}")
            save_log("cluster_bikes", "info", f"Załadowano model z {self.model_data['n_clusters']} klastrów oraz {self.model_data['silhouette_score']:.3f} silhouette score.")

            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            save_log("cluster_bikes", "error", f"Błąd ładowania modelu: {e}.")
            self.is_loaded = False
            return False

    # Przygotowuje cechy z pojedynczego dict'a (dla real-time predykcji)
    def prepare_features_from_dict(self, enriched_dict: Dict) -> Optional[pd.DataFrame]:
        
        # Mapowanie nazw kolumn z twojego enriched dict'a
        feature_mapping = {
            'bikes_available': 'bikes_available',
            'docks_available': 'docks_available', 
            'capacity': 'capacity',
            'manual_bikes_available': 'manual_bikes_available',
            'electric_bikes_available': 'electric_bikes_available',
            'temperature': 'temperature',
            'wind_kph': 'wind_kph',
            'precip_mm': 'precip_mm',
            'humidity': 'humidity',
            'weather_condition': 'weather_condition',
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10'
        }
        
        try:
            # Sprawdź czy wszystkie potrzebne klucze istnieją
            missing_keys = []
            feature_data = {}
            
            for model_key, dict_key in feature_mapping.items():
                if dict_key in enriched_dict:
                    feature_data[model_key] = enriched_dict[dict_key]
                else:
                    missing_keys.append(dict_key)
            
            if missing_keys:
                print(f"⚠️ Brakuje kluczy w danych: {missing_keys}")
                return None
            
            # Konweruj do DataFrame
            df = pd.DataFrame([feature_data])
            
            # Enkodowanie weather_condition z mapowaniem na 'unknown'
            try:
                df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                df = df.drop('weather_condition', axis=1)
            except ValueError as e:
                original_weather = df['weather_condition'].iloc[0]
                print(f"⚠️ Nieznana wartość weather_condition: '{original_weather}'")
                
                # Sprawdź czy 'unknown' istnieje w znanych klasach
                known_classes = list(self.label_encoder.classes_)
                if 'unknown' in known_classes:
                    print(f"🔄 Mapuję '{original_weather}' -> 'unknown'")
                    df['weather_condition'] = 'unknown'
                    df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    df = df.drop('weather_condition', axis=1)
                else:
                    print(f"❌ Brak klasy 'unknown' w modelu. Znane klasy: {known_classes}")
                    # Użyj pierwszej dostępnej klasy jako fallback
                    fallback_weather = known_classes[0]
                    print(f"🔄 Używam fallback: '{fallback_weather}'")
                    df['weather_condition'] = fallback_weather
                    df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    df = df.drop('weather_condition', axis=1)
            
            # Sprawdź czy są NaN
            if df.isnull().any().any():
                print("⚠️ Dane zawierają wartości NaN")
                return None
            
            save_log("cluster_bikes", "info", f"Pomyślnie przygotowano zmienne dla modelu z dicta.")
            return df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech: {e}")
            save_log("cluster_bikes", "error", f"Wystąpił błąd przy przygotowywaniu zmiennych dla modelu z dicta: {e}.")
            return None
    
    # Przewiduje klaster dla pojedynczego dict'a danych
    def predict_cluster_from_dict(self, enriched_dict: Dict) -> Optional[int]:
        
        if not self.is_loaded:
            print("⚠️ Model nie jest załadowany")
            return None
        
        # Przygotuj cechy
        features_df = self.prepare_features_from_dict(enriched_dict)
        
        if features_df is None:
            print("⚠️ Nie można przygotować cech")
            return None
        
        try:
            # Normalizacja
            features_scaled = self.scaler.transform(features_df)
            
            # Predykcja
            cluster_prediction = self.kmeans.predict(features_scaled)
            
            save_log("cluster_bikes", "info", f"Pomyślnie przewidziano klaster dla danych.")
            return int(cluster_prediction[0])
            
        except Exception as e:
            save_log("cluster_bikes", "error", f"Wystąpił błąd przy przewidywaniu klastra dla danych: {e}.")
            print(f"❌ Błąd predykcji klastra: {e}")
            return None
    
    # Zwraca informacje o centrum klastra
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        if not self.is_loaded or cluster_id >= len(self.kmeans.cluster_centers_):
            return None
        
        center = self.kmeans.cluster_centers_[cluster_id]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(center))]
        
        cluster_info = {}
        for feature, value in zip(feature_names, center):
            cluster_info[feature] = float(value)
        
        return cluster_info
    
    # Przeładowuje model z dysku
    def reload_model(self) -> bool:
        save_log(self.model_name, "info", f"🔄 Rozpoczynam przeładowywanie modelu '{self.model_name}'...")
        print(f"🔄 Przeładowywanie modelu '{self.model_name}'...")
        
        # Sprawdź, czy nowy plik modelu istnieje
        if not os.path.exists(self.new_model_path):
            save_log(self.model_name, "warning", f"Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            print(f"⚠️ Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            return False

        try:
            # 1. Usuń stary plik modelu (jeśli istnieje), aby zrobić miejsce na nowy
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
                save_log(self.model_name, "info", f"Usunięto stary plik modelu: {self.model_path}")
                print(f"Usunięto stary plik modelu: {self.model_path}")
            
            # 2. Zmień nazwę nowego pliku na "główny" plik modelu
            # Ta operacja jest atomowa na większości systemów plików.
            os.rename(self.new_model_path, self.model_path)
            save_log(self.model_name, "info", f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            print(f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            
            # 3. Załaduj nowo podmieniony model
            if self.load_model():
                save_log(self.model_name, "info", f"Model '{self.model_name}' pomyślnie przeładowany.")
                print(f"✅ Model '{self.model_name}' pomyślnie przeładowany.")
                return True
            else:
                # Jeśli ładowanie się nie powiodło po podmianie, to jest problem
                save_log(self.model_name, "error", f"Nie udało się załadować nowo podmienionego modelu '{self.model_name}'.")
                print(f"❌ Nie udało się załadować nowo podmienionego modelu '{self.model_name}'.")
                return False
                
        except Exception as e:
            save_log(self.model_name, "error", f"Błąd podczas atomowej podmiany modelu '{self.model_name}': {e}")
            print(f"❌ Błąd podczas atomowej podmiany modelu '{self.model_name}': {e}")
            self.is_loaded = False # Upewnij się, że flaga jest False w przypadku błędu
            return False

# +--------------------------------------------------+
# |      KLASA MODELU AUTOBUSOWE KLASTERYZACJI       |
# |             Funkcje do zarządzania               |
# +--------------------------------------------------+

"""
Klasa odpowiedzialna za predykcję klastra dla danych autobusowych.
Wykorzystuje wytrenowany model K-Means do grupowania autobusów/tras
o podobnych wzorcach opóźnień, uwzględniając zarówno czynniki operacyjne,
jak i środowiskowe.

Klasteryzowane cechy:
- Dane o opóźnieniach: 'average_delay_seconds', 'maximum_delay_seconds',
    'minimum_delay_seconds', 'delay_standard_deviation', 'delay_range_seconds',
    'on_time_stop_ratio', 'delay_consistency_score', 'stops_count'.
    Opisują charakterystykę opóźnień i punktualności, kluczowe dla optymalizacji
    rozkładów i zarządzania ruchem.
- Dane pogodowe: 'temperature', 'wind_kph', 'precip_mm', 'humidity',
    'weather_condition'.
    Warunki atmosferyczne mają istotny wpływ na ruch drogowy i punktualność autobusów.
- Dane o jakości powietrza: 'fine_particles_pm2_5', 'coarse_particles_pm10'.
    Jakość powietrza może być dodatkowym czynnikiem wpływającym na warunki ruchu.

Celem jest identyfikacja grup autobusów/tras o podobnych "profilach" opóźnień
w różnych warunkach, co wspiera diagnozowanie problemów i efektywne planowanie transportu.
"""

class BusClusterPredictor:
    # Inicjalizuje predyktor z wczytanym modelem
    def __init__(self, model_path='/app/shared/clusterization/models/buses_kmeans.pkl'):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace('.pkl', '') 
        self.new_model_path = model_path.replace('.pkl', '_new.pkl')
        self.model_data = None
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        # Przy inicjalizacji, jeśli istnieje "_new" plik, zastąp nim główny
        if os.path.exists(self.new_model_path):
            save_log(self.model_name, "info", f"Znaleziono nowy model {self.new_model_path} przy starcie. Przenoszę go na główną ścieżkę.")
            try:
                if os.path.exists(self.model_path):
                    os.remove(self.model_path) # Usuń stary model, jeśli istnieje
                os.rename(self.new_model_path, self.model_path) # Przemianuj nowy na główny
            except Exception as e:
                save_log(self.model_name, "error", f"Błąd przy przenoszeniu {self.new_model_path} na {self.model_path} podczas startu: {e}")
        
        self.load_model()
    
    # Wczytuje zapisany model i wszystkie komponenty
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model nie istnieje: {self.model_path}")
                return False
                
            print(f"🔄 Ładowanie modelu z: {self.model_path}")
            self.model_data = joblib.load(self.model_path)
            
            self.kmeans = self.model_data['kmeans']
            self.scaler = self.model_data['scaler']
            self.label_encoder = self.model_data['label_encoder']
            self.feature_names = self.model_data['feature_names']
            
            print(f"✅ Model załadowany pomyślnie!")
            print(f"   Liczba klastrów: {self.model_data['n_clusters']}")
            print(f"   Silhouette Score: {self.model_data['silhouette_score']:.3f}")
            save_log("cluster_buses", "info", f"Załadowano model z {self.model_data['n_clusters']} klastrów oraz {self.model_data['silhouette_score']:.3f} silhouette score.")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            save_log("cluster_buses", "error", f"Błąd ładowania modelu: {e}.")
            self.is_loaded = False
            return False

    # Przygotowuje cechy z pojedynczego dict'a (dla real-time predykcji)
    def prepare_features_from_dict(self, enriched_dict: Dict) -> Optional[pd.DataFrame]:
        
        # Mapowanie nazw kolumn z twojego enriched dict'a
        feature_mapping = {
            # Dane o opóźnieniach autobusów
            'average_delay_seconds': 'average_delay_seconds',
            'maximum_delay_seconds': 'maximum_delay_seconds',
            'minimum_delay_seconds': 'minimum_delay_seconds',
            'delay_standard_deviation': 'delay_standard_deviation',
            'delay_range_seconds': 'delay_range_seconds',
            'on_time_stop_ratio': 'on_time_stop_ratio',
            'delay_consistency_score': 'delay_consistency_score',
            'stops_count': 'stops_count',
            'temperature': 'temperature',
            'wind_kph': 'wind_kph',
            'precip_mm': 'precip_mm',
            'humidity': 'humidity',
            'weather_condition': 'weather_condition',
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10'
        }
        
        try:
            # Sprawdź czy wszystkie potrzebne klucze istnieją
            missing_keys = []
            feature_data = {}
            
            for model_key, dict_key in feature_mapping.items():
                if dict_key in enriched_dict:
                    feature_data[model_key] = enriched_dict[dict_key]
                else:
                    missing_keys.append(dict_key)
            
            if missing_keys:
                print(f"⚠️ Brakuje kluczy w danych: {missing_keys}")
                return None
            
            # Konweruj do DataFrame
            df = pd.DataFrame([feature_data])
            
            # Enkodowanie weather_condition z mapowaniem na 'unknown'
            try:
                df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                df = df.drop('weather_condition', axis=1)
            except ValueError as e:
                original_weather = df['weather_condition'].iloc[0]
                print(f"⚠️ Nieznana wartość weather_condition: '{original_weather}'")
                
                # Sprawdź czy 'unknown' istnieje w znanych klasach
                known_classes = list(self.label_encoder.classes_)
                if 'unknown' in known_classes:
                    print(f"🔄 Mapuję '{original_weather}' -> 'unknown'")
                    df['weather_condition'] = 'unknown'
                    df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    df = df.drop('weather_condition', axis=1)
                else:
                    print(f"❌ Brak klasy 'unknown' w modelu. Znane klasy: {known_classes}")
                    # Użyj pierwszej dostępnej klasy jako fallback
                    fallback_weather = known_classes[0]
                    print(f"🔄 Używam fallback: '{fallback_weather}'")
                    df['weather_condition'] = fallback_weather
                    df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    df = df.drop('weather_condition', axis=1)
            
            # Sprawdź czy są NaN
            if df.isnull().any().any():
                print("⚠️ Dane zawierają wartości NaN")
                return None
            
            save_log("cluster_buses", "info", f"Pomyślnie przygotowano zmienne dla modelu z dicta.")
            return df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech: {e}")
            save_log("cluster_buses", "error", f"Wystąpił błąd przy przygotowywaniu zmiennych dla modelu z dicta: {e}.")
            return None
    
    # Przewiduje klaster dla pojedynczego dict'a danych
    def predict_cluster_from_dict(self, enriched_dict: Dict) -> Optional[int]:
        
        if not self.is_loaded:
            print("⚠️ Model nie jest załadowany")
            return None
        
        # Przygotuj cechy
        features_df = self.prepare_features_from_dict(enriched_dict)
        
        if features_df is None:
            print("⚠️ Nie można przygotować cech")
            return None
        
        try:
            # Normalizacja
            features_scaled = self.scaler.transform(features_df)
            
            # Predykcja
            cluster_prediction = self.kmeans.predict(features_scaled)
            
            save_log("cluster_buses", "info", f"Pomyślnie przewidziano klaster dla danych.")
            return int(cluster_prediction[0])
            
        except Exception as e:
            print(f"❌ Błąd predykcji klastra: {e}")
            save_log("cluster_buses", "error", f"Wystąpił błąd przy przewidywaniu klastra dla danych: {e}.")
            return None
    
    # Zwraca informacje o centrum klastra
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        if not self.is_loaded or cluster_id >= len(self.kmeans.cluster_centers_):
            return None
        
        center = self.kmeans.cluster_centers_[cluster_id]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(center))]
        
        cluster_info = {}
        for feature, value in zip(feature_names, center):
            cluster_info[feature] = float(value)
        
        return cluster_info
    
    # Przeładowuje model z dysku
    def reload_model(self) -> bool:
        save_log(self.model_name, "info", f"🔄 Rozpoczynam przeładowywanie modelu '{self.model_name}'...")
        print(f"🔄 Przeładowywanie modelu '{self.model_name}'...")
        
        # Sprawdź, czy nowy plik modelu istnieje
        if not os.path.exists(self.new_model_path):
            save_log(self.model_name, "warning", f"Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            print(f"⚠️ Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            return False

        try:
            # 1. Usuń stary plik modelu (jeśli istnieje), aby zrobić miejsce na nowy
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
                save_log(self.model_name, "info", f"Usunięto stary plik modelu: {self.model_path}")
                print(f"Usunięto stary plik modelu: {self.model_path}")
            
            # 2. Zmień nazwę nowego pliku na "główny" plik modelu
            # Ta operacja jest atomowa na większości systemów plików.
            os.rename(self.new_model_path, self.model_path)
            save_log(self.model_name, "info", f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            print(f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            
            # 3. Załaduj nowo podmieniony model
            if self.load_model():
                save_log(self.model_name, "info", f"Model '{self.model_name}' pomyślnie przeładowany.")
                print(f"✅ Model '{self.model_name}' pomyślnie przeładowany.")
                return True
            else:
                # Jeśli ładowanie się nie powiodło po podmianie, to jest problem
                save_log(self.model_name, "error", f"Nie udało się załadować nowo podmienionego modelu '{self.model_name}'.")
                print(f"❌ Nie udało się załadować nowo podmienionego modelu '{self.model_name}'.")
                return False
                
        except Exception as e:
            save_log(self.model_name, "error", f"Błąd podczas atomowej podmiany modelu '{self.model_name}': {e}")
            print(f"❌ Błąd podczas atomowej podmiany modelu '{self.model_name}': {e}")
            self.is_loaded = False # Upewnij się, że flaga jest False w przypadku błędu
            return False







# +--------------------------------------------------+
# |     FUNKCJE DODATKOWE DO ZARZĄDZANIA MODELAMI    |
# |             Funkcje do zarządzania               |
# +--------------------------------------------------+

# Globalne instancje predyktorów dla łatwego dostępu
# Będą ładowane przy pierwszym imporcie tego pliku
bike_cluster_predictor = BikeStationClusterPredictor(model_path='/app/shared/clusterization/models/bikes_kmeans.pkl')
bus_cluster_predictor = BusClusterPredictor(model_path='/app/shared/clusterization/models/buses_kmeans.pkl')

# Przeładowuje wszystkie modele z dysku
def reload_models():

    global bike_cluster_predictor, bus_cluster_predictor
    
    print("🔄 Przeładowywanie modeli klastrów...")
    
    if bike_cluster_predictor:
        bike_cluster_predictor.reload_model()
    
    if bus_cluster_predictor:
        bus_cluster_predictor.reload_model()
    
    print("✅ Przeładowywanie modeli klastrów zakończone")
    save_log("cluster_module", "info", f"Przeładowano wszystkie modele klasteryzacji.")

# Zwraca status wszystkich modeli
def get_models_status() -> Dict:

    global bike_cluster_predictor, bus_cluster_predictor
    
    return {
        'bike_cluster_predictor': {
            'loaded': bike_cluster_predictor.is_loaded,
            'model_path': bike_cluster_predictor.model_path,
            'n_clusters': bike_cluster_predictor.model_data.get('n_clusters') if bike_cluster_predictor.is_loaded else None
        },
        'bus_cluster_predictor': {
            'loaded': bus_cluster_predictor.is_loaded,
            'model_path': bus_cluster_predictor.model_path,
            'n_clusters': bus_cluster_predictor.model_data.get('n_clusters') if bus_cluster_predictor.is_loaded else None
        }
    }