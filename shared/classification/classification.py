import pandas as pd
import joblib
import os
from typing import Dict, Optional, Union, Tuple, List

# Bazowa ścieżka do katalogu z modelami
# Zmień tę ścieżkę, jeśli Twoje modele są w innym miejscu
BASE_MODEL_PATH = '/app/shared/classification/models/'

# --- Mapowania tekstowe dla klasyfikacji ---
IS_LATE_MAPPING = {
    0: "on time",
    1: "late"
}

DELAY_CATEGORY_MAPPING = {
    0: "on time",
    1: "slightly late",
    2: "very late"
}

BIKE_BINARY_MAPPING = {
    0: "sufficient",
    1: "low"
}

BIKE_MULTICLASS_MAPPING = {
    0: "none",
    1: "low availability",
    2: "moderate availability",
    3: "high availability"
}


class BasePredictor:
    """
    Bazowa klasa dla predyktorów modeli, obsługująca ładowanie i status.
    """
    def __init__(self, model_type: str, data_source: str, model_name: str):
        self.model_type = model_type  # np. 'binary', 'multiclass', 'regression'
        self.data_source = data_source  # np. 'bus', 'bike'
        self.model_name = model_name  # np. 'bus_binary_model.pkl'
        self.model_path = os.path.join(BASE_MODEL_PATH, model_name)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None # Używane tylko w przypadku klasyfikacji z enkodowaniem etykiet
        self.feature_names = None
        self.is_loaded = False
        self.load_status_message = "Model nie został jeszcze załadowany."

        self.load_model()

    def load_model(self) -> bool:
        """
        Wczytuje zapisany model i wszystkie komponenty (model, scaler, feature_names).
        """
        try:
            if not os.path.exists(self.model_path):
                self.load_status_message = f"⚠️ Model nie istnieje: {self.model_path}"
                print(self.load_status_message)
                self.is_loaded = False
                return False
                
            print(f"🔄 Ładowanie modelu z: {self.model_path}")
            model_data = joblib.load(self.model_path)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder') # Może nie istnieć dla wszystkich modeli
            self.feature_names = model_data.get('feature_names')
            
            if self.model is None or self.scaler is None or self.feature_names is None:
                self.load_status_message = f"❌ Błąd: Brak kluczowych komponentów (model, scaler, feature_names) w pliku {self.model_path}"
                print(self.load_status_message)
                self.is_loaded = False
                return False

            self.is_loaded = True
            self.load_status_message = f"✅ Model '{self.model_name}' załadowany pomyślnie!"
            print(self.load_status_message)
            return True
            
        except Exception as e:
            self.load_status_message = f"❌ Błąd ładowania modelu '{self.model_name}': {e}"
            print(self.load_status_message)
            self.is_loaded = False
            return False

    def get_status(self) -> Dict:
        """
        Zwraca status załadowania modelu.
        """
        return {
            'loaded': self.is_loaded,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'data_source': self.data_source,
            'status_message': self.load_status_message,
            'feature_names_count': len(self.feature_names) if self.feature_names else 0
        }

    def reload_model(self) -> bool:
        """
        Przeładowuje model z dysku.
        """
        print(f"🔄 Przeładowywanie modelu {self.model_name}...")
        return self.load_model()

    def _prepare_features(self, data_dict: Dict, feature_mapping: Dict) -> Optional[pd.DataFrame]:
        """
        Wewnętrzna metoda do przygotowywania cech z dict'a.
        """
        if not self.is_loaded:
            print(f"⚠️ Model '{self.model_name}' nie jest załadowany. Nie można przygotować cech.")
            return None

        feature_data = {}
        missing_keys = []

        for model_key, dict_key in feature_mapping.items():
            if dict_key in data_dict:
                feature_data[model_key] = data_dict[dict_key]
            else:
                missing_keys.append(dict_key)
        
        if missing_keys:
            print(f"⚠️ Brakuje kluczy w danych dla modelu '{self.model_name}': {missing_keys}")
            return None

        try:
            df = pd.DataFrame([feature_data])

            # Obsługa 'daylight' jeśli istnieje i jest w formatcie 'yes'/'no'
            if 'daylight' in df.columns and df['daylight'].dtype == 'object':
                if 'daylight' in self.feature_names: # Sprawdź czy model oczekuje 'daylight' jako numeryczne
                    df['daylight'] = df['daylight'].map({'yes': 1, 'no': 0}).fillna(0) # Użyj 0 jako fallback
                    if df['daylight'].isnull().any():
                        print(f"⚠️ Nieznana wartość 'daylight' dla modelu '{self.model_name}'. Użyto 0.")

            # Obsługa 'weather_condition' jeśli istnieje i jest enkodowana
            if 'weather_condition' in df.columns and 'weather_condition_encoded' in self.feature_names:
                if self.label_encoder:
                    original_weather = df['weather_condition'].iloc[0]
                    try:
                        df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    except ValueError:
                        print(f"⚠️ Nieznana wartość weather_condition: '{original_weather}' dla modelu '{self.model_name}'")
                        known_classes = list(self.label_encoder.classes_)
                        if 'unknown' in known_classes:
                            print(f"🔄 Mapuję '{original_weather}' -> 'unknown'")
                            df['weather_condition'] = 'unknown'
                            df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                        elif known_classes: # Fallback do pierwszej znanej klasy
                            fallback_weather = known_classes[0]
                            print(f"🔄 Brak klasy 'unknown'. Używam fallback: '{fallback_weather}'")
                            df['weather_condition'] = fallback_weather
                            df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                        else:
                            print(f"❌ Brak znanych klas dla 'weather_condition' w modelu '{self.model_name}'. Nie można zakodować.")
                            return None
                    df = df.drop('weather_condition', axis=1)
                else:
                    print(f"⚠️ Brak LabelEncoder dla 'weather_condition' w modelu '{self.model_name}'. Pomijam enkodowanie.")
                    # Jeśli model oczekuje 'weather_condition_encoded' ale nie ma encodera, to jest problem
                    if 'weather_condition_encoded' in self.feature_names:
                        print(f"❌ Model '{self.model_name}' oczekuje 'weather_condition_encoded' ale brak LabelEncoder. Predykcja niemożliwa.")
                        return None
                    else: # Jeśli model nie oczekuje zakodowanego, to po prostu usuń oryginalną kolumnę
                        df = df.drop('weather_condition', axis=1)
            elif 'weather_condition' in df.columns: # Jeśli weather_condition istnieje, ale model nie oczekuje encoded
                 df = df.drop('weather_condition', axis=1) # Usuń, jeśli nie jest potrzebna

            # Upewnij się, że DataFrame ma te same kolumny i w tej samej kolejności co podczas treningu
            # Dodaj brakujące kolumny z wartością 0 (jeśli to cechy binarne/kategoryczne po One-Hot Encoding)
            # Lub usuń nadmiarowe kolumny
            processed_df = pd.DataFrame(columns=self.feature_names)
            for col in self.feature_names:
                if col in df.columns:
                    processed_df[col] = df[col]
                else:
                    processed_df[col] = 0 # Domyślna wartość dla brakujących cech (np. po One-Hot Encoding)
            
            if processed_df.isnull().any().any():
                print(f"⚠️ Dane po przygotowaniu dla modelu '{self.model_name}' zawierają wartości NaN.")
                return None
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech dla modelu '{self.model_name}': {e}")
            return None

class BusModelPredictor(BasePredictor):
    """
    Klasa do predykcji dla modeli autobusowych (binarny, wieloklasowy, regresja).
    """
    def __init__(self, model_type: str, model_name: str):
        super().__init__(model_type, 'bus', model_name)
        # Mapowanie cech dla danych autobusowych
        self.feature_mapping = {
            'stops_count': 'stops_count',
            'maximum_delay_seconds': 'maximum_delay_seconds',
            'minimum_delay_seconds': 'minimum_delay_seconds',
            'delay_variance_value': 'delay_variance_value',
            'delay_standard_deviation': 'delay_standard_deviation',
            'delay_range_seconds': 'delay_range_seconds',
            'stops_on_time_count': 'stops_on_time_count',
            'stops_arrived_early_count': 'stops_arrived_early_count',
            'stops_arrived_late_count': 'stops_arrived_late_count',
            'delay_consistency_score': 'delay_consistency_score',
            'on_time_stop_ratio': 'on_time_stop_ratio',
            'avg_positive_delay_seconds': 'avg_positive_delay_seconds',
            'avg_negative_delay_seconds': 'avg_negative_delay_seconds',
            'temperature': 'temperature',
            'feelslike': 'feelslike',
            'humidity': 'humidity',
            'wind_kph': 'wind_kph',
            'precip_mm': 'precip_mm',
            'cloud': 'cloud',
            'visibility_km': 'visibility_km',
            'uv_index': 'uv_index',
            'daylight': 'daylight',
            'weather_condition': 'weather_condition', # Będzie enkodowane jeśli model tego wymaga
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10',
            'carbon_monoxide_ppb': 'carbon_monoxide_ppb',
            'nitrogen_dioxide_ppb': 'nitrogen_dioxide_ppb',
            'ozone_ppb': 'ozone_ppb',
            'sulfur_dioxide_ppb': 'sulfur_dioxide_ppb',
            'cluster_id': 'cluster_id'
        }

    def predict(self, data_dict: Dict) -> Optional[Union[float, Tuple[int, List[float], str]]]:
        """
        Wykonuje predykcję na podstawie słownika danych wejściowych.
        Zwraca przewidywaną wartość/klasę lub (klasę numeryczną, prawdopodobieństwa, klasę tekstową) dla klasyfikacji.
        """
        if not self.is_loaded:
            print(f"⚠️ Model '{self.model_name}' nie jest załadowany. Nie można wykonać predykcji.")
            return None
        
        # Przygotuj cechy
        features_df = self._prepare_features(data_dict, self.feature_mapping)
        
        if features_df is None:
            print(f"⚠️ Nie można przygotować cech dla modelu '{self.model_name}'.")
            return None
        
        try:
            # Normalizacja danych
            data_scaled = self.scaler.transform(features_df)
            
            # Predykcja
            if self.model_type == 'binary':
                prediction_num = self.model.predict(data_scaled)[0]
                probabilities = self.model.predict_proba(data_scaled)[0].tolist()
                prediction_label = IS_LATE_MAPPING.get(prediction_num, "nieznany_status")
                return int(prediction_num), probabilities, prediction_label
            elif self.model_type == 'multiclass':
                prediction_num = self.model.predict(data_scaled)[0]
                probabilities = self.model.predict_proba(data_scaled)[0].tolist()
                prediction_label = DELAY_CATEGORY_MAPPING.get(prediction_num, "nieznana_kategoria")
                return int(prediction_num), probabilities, prediction_label
            elif self.model_type == 'regression':
                prediction = self.model.predict(data_scaled)[0]
                return float(prediction)
            else:
                print(f"❌ Nieznany typ modelu: {self.model_type}")
                return None
                
        except Exception as e:
            print(f"❌ Błąd predykcji dla modelu '{self.model_name}': {e}")
            return None


class BikeModelPredictor(BasePredictor):
    """
    Klasa do predykcji dla modeli stacji rowerowych (binarny, wieloklasowy, regresja).
    """
    def __init__(self, model_type: str, model_name: str):
        super().__init__(model_type, 'bike', model_name)
        # Mapowanie cech dla danych stacji rowerowych
        self.feature_mapping = {
            'bikes_available': 'bikes_available',
            'docks_available': 'docks_available', 
            'capacity': 'capacity',
            'manual_bikes_available': 'manual_bikes_available',
            'electric_bikes_available': 'electric_bikes_available',
            'temperature': 'temperature',
            'wind_kph': 'wind_kph',
            'precip_mm': 'precip_mm',
            'humidity': 'humidity',
            'weather_condition': 'weather_condition', # Będzie enkodowane jeśli model tego wymaga
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10',
            'cluster_id': 'cluster_id' # Zakładam, że cluster_id jest również cechą dla modeli rowerowych
        }

    def predict(self, data_dict: Dict) -> Optional[Union[float, Tuple[int, List[float], str]]]:
        """
        Wykonuje predykcję na podstawie słownika danych wejściowych.
        Zwraca przewidywaną wartość/klasę lub (klasę numeryczną, prawdopodobieństwa, klasę tekstową) dla klasyfikacji.
        """
        if not self.is_loaded:
            print(f"⚠️ Model '{self.model_name}' nie jest załadowany. Nie można wykonać predykcji.")
            return None
        
        # Przygotuj cechy
        features_df = self._prepare_features(data_dict, self.feature_mapping)
        
        if features_df is None:
            print(f"⚠️ Nie można przygotować cech dla modelu '{self.model_name}'.")
            return None
        
        try:
            # Normalizacja danych
            data_scaled = self.scaler.transform(features_df)
            
            # Predykcja
            if self.model_type == 'binary':
                prediction_num = self.model.predict(data_scaled)[0]
                probabilities = self.model.predict_proba(data_scaled)[0].tolist()
                prediction_label = BIKE_BINARY_MAPPING.get(prediction_num, "nieznany_status_rower")
                return int(prediction_num), probabilities, prediction_label
            elif self.model_type == 'multiclass':
                prediction_num = self.model.predict(data_scaled)[0]
                probabilities = self.model.predict_proba(data_scaled)[0].tolist()
                prediction_label = BIKE_MULTICLASS_MAPPING.get(prediction_num, "nieznana_kategoria_rower")
                return int(prediction_num), probabilities, prediction_label
            elif self.model_type == 'regression':
                prediction = self.model.predict(data_scaled)[0]
                return float(prediction)
            else:
                print(f"❌ Nieznany typ modelu: {self.model_type}")
                return None
                
        except Exception as e:
            print(f"❌ Błąd predykcji dla modelu '{self.model_name}': {e}")
            return None


# Globalne instancje predyktorów dla łatwego dostępu
# Będą ładowane przy pierwszym imporcie tego pliku
bus_binary_predictor = BusModelPredictor('binary', 'bus_binary_model.pkl')
bus_multiclass_predictor = BusModelPredictor('multiclass', 'bus_multiclass_model.pkl')
bus_regression_predictor = BusModelPredictor('regression', 'bus_regression_model.pkl')

# Zakładam, że masz również analogiczne modele dla rowerów
# Jeśli nie masz tych plików .pkl, te linie wygenerują błędy ładowania,
# ale klasy są gotowe do ich obsługi, gdy tylko modele będą dostępne.
bike_binary_predictor = BikeModelPredictor('binary', 'bike_binary_model.pkl')
bike_multiclass_predictor = BikeModelPredictor('multiclass', 'bike_multiclass_model.pkl')
bike_regression_predictor = BikeModelPredictor('regression', 'bike_regression_model.pkl')


def reload_all_predictors() -> Dict:
    """
    Przeładowuje wszystkie globalne instancje predyktorów.
    """
    print("🔄 Przeładowywanie wszystkich predyktorów...")
    results = {}
    
    for predictor in [
        bus_binary_predictor, bus_multiclass_predictor, bus_regression_predictor,
        bike_binary_predictor, bike_multiclass_predictor, bike_regression_predictor
    ]:
        if predictor:
            reloaded = predictor.reload_model()
            results[predictor.model_name] = "Przeładowano pomyślnie" if reloaded else "Błąd przeładowywania"
    
    print("✅ Przeładowywanie wszystkich predyktorów zakończone.")
    return results

def get_all_predictors_status() -> Dict:
    """
    Zwraca status załadowania dla wszystkich globalnych predyktorów.
    """
    status = {}
    for predictor in [
        bus_binary_predictor, bus_multiclass_predictor, bus_regression_predictor,
        bike_binary_predictor, bike_multiclass_predictor, bike_regression_predictor
    ]:
        if predictor:
            status[predictor.model_name] = predictor.get_status()
    return status
