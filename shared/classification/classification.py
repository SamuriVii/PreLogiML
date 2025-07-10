from typing import Dict, Optional, Union, Tuple, List
from shared.db_utils import save_log
import pandas as pd
import joblib
import os

# Bazowa ścieżka do katalogu z modelami
BASE_MODEL_PATH = '/app/shared/classification/models/'

# +--------------------------------------------------+
# |      MAPOWANIE TEKSTOWE DLA KLASYFIAKTORÓW       |
# |                   Słowniki                       |
# +--------------------------------------------------+

# Dane autobusowe - słowniki
IS_LATE_MAPPING = {
    0: "na czas",
    1: "spóźniony"
}

DELAY_CATEGORY_MAPPING = {
    0: "na czas",
    1: "lekko spóźniony",
    2: "bardzo spóźniony"
}

# Dane rowerowe - słowniki
BIKE_BINARY_MAPPING = {
    0: "standardowo",
    1: "mało"
}

BIKE_MULTICLASS_MAPPING = {
    0: "brak",
    1: "mała dostępność",
    2: "standardowa dostępność",
    3: "wysoka dostępność"
}

# +--------------------------------------------------+
# |    PRZYGOTOWANIE DANYCH I ZARZĄDZANIE MODELAM    |
# |            Funkcja do zarządzania                |
# +--------------------------------------------------+

# Bazowa klasa dla predyktorów modeli, obsługująca ładowanie i status.
class BasePredictor:
    def __init__(self, model_type: str, data_source: str, model_name: str):
        self.model_type = model_type 
        self.data_source = data_source
        self.raw_model_name = model_name
        self.model_path = os.path.join(BASE_MODEL_PATH, model_name)
        self.new_model_path = self.model_path.replace('.pkl', '_new.pkl')

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        self.load_status_message = "Model nie został jeszcze załadowany."

        # Przy inicjalizacji, jeśli istnieje "_new" plik, zastąp nim główny
        if os.path.exists(self.new_model_path):
            log_identifier = f"{self.data_source}_{self.model_type}_predictor"
            save_log(log_identifier, "info", f"Znaleziono nowy model {self.new_model_path} przy starcie. Przenoszę go na główną ścieżkę.")
            print(f"Znaleziono nowy model {self.new_model_path} przy starcie. Przenoszę go na główną ścieżkę.")
            try:
                if os.path.exists(self.model_path):
                    os.remove(self.model_path) # Usuń stary model, jeśli istnieje
                    save_log(log_identifier, "info", f"Usunięto stary plik modelu: {self.model_path}")
                    print(f"Usunięto stary plik modelu: {self.model_path}")
                os.rename(self.new_model_path, self.model_path) # Przemianuj nowy na główny
                save_log(log_identifier, "info", f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
                print(f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            except Exception as e:
                save_log(log_identifier, "error", f"Błąd przy przenoszeniu {self.new_model_path} na {self.model_path} podczas startu: {e}")
                print(f"Błąd przy przenoszeniu {self.new_model_path} na {self.model_path} podczas startu: {e}")
        
        self.load_model()

    # Wczytuje zapisany model i wszystkie komponenty (model, scaler, feature_names).
    def load_model(self) -> bool:

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
            self.label_encoder = model_data.get('label_encoder')
            self.feature_names = model_data.get('feature_names')
            
            if self.model is None or self.scaler is None or self.feature_names is None:
                self.load_status_message = f"❌ Błąd: Brak kluczowych komponentów (model, scaler, feature_names) w pliku {self.model_path}"
                print(self.load_status_message)
                self.is_loaded = False
                return False

            self.is_loaded = True
            self.load_status_message = f"✅ Model '{self.raw_model_name}' załadowany pomyślnie!"
            save_log("class_module", "info", "Model klasyfikacji został załadowany pomyślnie")
            print(self.load_status_message)
            return True
            
        except Exception as e:
            self.load_status_message = f"❌ Błąd ładowania modelu '{self.raw_model_name}': {e}"
            save_log("class_module", "erro", f"Wystąpił błąd przy ładowaniu modelu klasyfikacji: {e}.")
            print(self.load_status_message)
            self.is_loaded = False
            return False

    # Zwraca status modelu
    def get_status(self) -> Dict:
        return {
            'loaded': self.is_loaded,
            'model_path': self.model_path,
            'raw_model_name': self.raw_model_name,
            'model_type': self.model_type,
            'data_source': self.data_source,
            'status_message': self.load_status_message,
            'feature_names_count': len(self.feature_names) if self.feature_names else 0
        }

    # Przeładowuje model z dysku
    def reload_model(self) -> bool:
        log_identifier = f"{self.data_source}_{self.model_type}_predictor"
        save_log(log_identifier, "info", f"🔄 Rozpoczynam przeładowywanie modelu '{self.raw_model_name}'...")
        print(f"🔄 Przeładowywanie modelu '{self.raw_model_name}'...")
        
        # Sprawdź, czy nowy plik modelu istnieje
        if not os.path.exists(self.new_model_path):
            save_log(log_identifier, "warning", f"Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            print(f"⚠️ Brak nowego pliku modelu do przeładowania: {self.new_model_path}")
            return False

        try:
            # 1. Usuń stary plik modelu (jeśli istnieje), aby zrobić miejsce na nowy
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
                save_log(log_identifier, "info", f"Usunięto stary plik modelu: {self.model_path}")
                print(f"Usunięto stary plik modelu: {self.model_path}")
            
            # 2. Zmień nazwę nowego pliku na "główny" plik modelu
            # Ta operacja jest atomowa na większości systemów plików.
            os.rename(self.new_model_path, self.model_path)
            save_log(log_identifier, "info", f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            print(f"Zmieniono nazwę {self.new_model_path} na {self.model_path}.")
            
            # 3. Załaduj nowo podmieniony model
            if self.load_model():
                save_log(log_identifier, "info", f"Model '{self.raw_model_name}' pomyślnie przeładowany.")
                print(f"✅ Model '{self.raw_model_name}' pomyślnie przeładowany.")
                return True
            else:
                # Jeśli ładowanie się nie powiodło po podmianie, to jest problem
                save_log(log_identifier, "error", f"Nie udało się załadować nowo podmienionego modelu '{self.raw_model_name}'.")
                print(f"❌ Nie udało się załadować nowo podmienionego modelu '{self.raw_model_name}'.")
                return False
                
        except Exception as e:
            save_log(log_identifier, "error", f"Błąd podczas atomowej podmiany modelu '{self.raw_model_name}': {e}")
            print(f"❌ Błąd podczas atomowej podmiany modelu '{self.raw_model_name}': {e}")
            self.is_loaded = False # Upewnij się, że flaga jest False w przypadku błędu
            return False

    # Wewnętrzna metoda do przygotowywania cech z dict'a.
    def _prepare_features(self, data_dict: Dict, feature_mapping: Dict) -> Optional[pd.DataFrame]:
        if not self.is_loaded:
            print(f"⚠️ Model '{self.raw_model_name}' nie jest załadowany. Nie można przygotować cech.")
            return None

        feature_data = {}
        missing_keys = []

        for model_key, dict_key in feature_mapping.items():
            if dict_key in data_dict:
                feature_data[model_key] = data_dict[dict_key]
            else:
                missing_keys.append(dict_key)
        
        if missing_keys:
            print(f"⚠️ Brakuje kluczy w danych dla modelu '{self.raw_model_name}': {missing_keys}")
            return None

        try:
            df = pd.DataFrame([feature_data])

            # Obsługa 'daylight' jeśli istnieje i jest w formatcie 'yes'/'no'
            if 'daylight' in df.columns and df['daylight'].dtype == 'object':
                if 'daylight' in self.feature_names: # Sprawdź czy model oczekuje 'daylight' jako numeryczne
                    df['daylight'] = df['daylight'].map({'yes': 1, 'no': 0}).fillna(0) # Użyj 0 jako fallback
                    if df['daylight'].isnull().any():
                        print(f"⚠️ Nieznana wartość 'daylight' dla modelu '{self.raw_model_name}'. Użyto 0.")

            # Obsługa 'weather_condition' jeśli istnieje i jest enkodowana
            if 'weather_condition' in df.columns and 'weather_condition_encoded' in self.feature_names:
                if self.label_encoder:
                    original_weather = df['weather_condition'].iloc[0]
                    try:
                        df['weather_condition_encoded'] = self.label_encoder.transform(df['weather_condition'])
                    except ValueError:
                        print(f"⚠️ Nieznana wartość weather_condition: '{original_weather}' dla modelu '{self.raw_model_name}'")
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
                            print(f"❌ Brak znanych klas dla 'weather_condition' w modelu '{self.raw_model_name}'. Nie można zakodować.")
                            return None
                    df = df.drop('weather_condition', axis=1)
                else:
                    print(f"⚠️ Brak LabelEncoder dla 'weather_condition' w modelu '{self.raw_model_name}'. Pomijam enkodowanie.")
                    # Jeśli model oczekuje 'weather_condition_encoded' ale nie ma encodera, to jest problem
                    if 'weather_condition_encoded' in self.feature_names:
                        print(f"❌ Model '{self.raw_model_name}' oczekuje 'weather_condition_encoded' ale brak LabelEncoder. Predykcja niemożliwa.")
                        return None
                    else: # Jeśli model nie oczekuje zakodowanego, to po prostu usuń oryginalną kolumnę
                        df = df.drop('weather_condition', axis=1)
            elif 'weather_condition' in df.columns: # Jeśli weather_condition istnieje, ale model nie oczekuje encoded
                 df = df.drop('weather_condition', axis=1) # Usuń, jeśli nie jest potrzebna

            # Upewnij się, że DataFrame ma te same kolumny i w tej samej kolejności co podczas treningu oraz dostosuj je do tamtego układu
            processed_df = pd.DataFrame(columns=self.feature_names)
            for col in self.feature_names:
                if col in df.columns:
                    processed_df[col] = df[col]
                else:
                    processed_df[col] = 0
            
            processed_df = df.reindex(columns=self.feature_names, fill_value=0)

            if processed_df.isnull().any().any():
                print(f"⚠️ Dane po przygotowaniu dla modelu '{self.raw_model_name}' zawierają wartości NaN.")
                return None
            save_log("class_module", "info", "Dane zostały przygotowane dla modelu klasyfikacji.")
            return processed_df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech dla modelu '{self.raw_model_name}': {e}")
            return None

# +--------------------------------------------------+
# |         PRZYGOTOWANIE DANYCH AUTOBUSOWYCH        |
# |             Funkcje do zarządzania               |
# +--------------------------------------------------+

"""
Klasa odpowiedzialna za predykcję różnych aspektów dotyczących autobusów.
Wykorzystuje wytrenowane modele klasyfikacji (binarnej, wieloklasowej)
lub regresji do przewidywania opóźnień autobusów na podstawie wielu cech.

Klasteryzowane cechy:
- Dane o opóźnieniach i punktualności: 'stops_count', 'maximum_delay_seconds',
    'minimum_delay_seconds', 'delay_variance_value', 'delay_standard_deviation',
    'delay_range_seconds', 'stops_on_time_count', 'stops_arrived_early_count',
    'stops_arrived_late_count', 'delay_consistency_score', 'on_time_stop_ratio',
    'avg_positive_delay_seconds', 'avg_negative_delay_seconds'.
    Te cechy opisują złożoność trasy oraz historyczne i bieżące wskaźniki opóźnień,
    pozwalając modelowi zrozumieć charakterystykę ruchu autobusowego.
- Dane pogodowe: 'temperature', 'feelslike', 'humidity', 'wind_kph',
    'precip_mm', 'cloud', 'visibility_km', 'uv_index', 'daylight', 'weather_condition'.
    Warunki pogodowe są kluczowymi czynnikami wpływającymi na ruch drogowy i punktualność
    transportu publicznego, stąd ich uwzględnienie pozwala na bardziej precyzyjne predykcje.
- Dane o jakości powietrza: 'fine_particles_pm2_5', 'coarse_particles_pm10',
    'carbon_monoxide_ppb', 'nitrogen_dioxide_ppb', 'ozone_ppb', 'sulfur_dioxide_ppb'.
    Zanieczyszczenie powietrza może pośrednio wpływać na warunki drogowe lub decyzje
    operacyjne, co czyni je ważnym kontekstowym elementem predykcji.
- 'cluster_id': Identyfikator klastra, do którego należy dany punkt danych,
    pochodzący z wcześniej przeprowadzonej klasteryzacji. Daje to modelowi
    dodatkową informację kontekstową o typowym zachowaniu danej "grupy" autobusów/tras.

Celem jest przewidywanie opóźnień autobusów (binarnie: na czas/spóźniony; wieloklasowo:
na czas/nieznaczne opóźnienie/duże opóźnienie; regresja: przewidywana wartość opóźnienia)
w oparciu o kompleksowy zestaw danych.
"""

class BusModelPredictor(BasePredictor):

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
            'weather_condition': 'weather_condition',
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10',
            'carbon_monoxide_ppb': 'carbon_monoxide_ppb',
            'nitrogen_dioxide_ppb': 'nitrogen_dioxide_ppb',
            'ozone_ppb': 'ozone_ppb',
            'sulfur_dioxide_ppb': 'sulfur_dioxide_ppb',
            'cluster_id': 'cluster_id'
        }

    # Wykonuje predykcję na podstawie słownika danych wejściowych.
    # Zwraca przewidywaną wartość/klasę lub (klasę numeryczną, prawdopodobieństwa, klasę tekstową) dla klasyfikacji.
    def predict(self, data_dict: Dict) -> Optional[Union[float, Tuple[int, List[float], str]]]:
        if not self.is_loaded:
            print(f"⚠️ Model '{self.raw_model_name}' nie jest załadowany. Nie można wykonać predykcji.")
            return None
        
        # Przygotuj cechy
        features_df = self._prepare_features(data_dict, self.feature_mapping)
        
        if features_df is None:
            print(f"⚠️ Nie można przygotować cech dla modelu '{self.raw_model_name}'.")
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
            print(f"❌ Błąd predykcji dla modelu '{self.raw_model_name}': {e}")
            return None

# +--------------------------------------------------+
# |         PRZYGOTOWANIE DANYCH ROWEROWYCH          |
# |             Funkcje do zarządzania               |
# +--------------------------------------------------+

"""
Klasa odpowiedzialna za predykcję dostępności rowerów na stacjach.
Wykorzystuje wytrenowane modele klasyfikacji (binarnej, wieloklasowej)
lub regresji do przewidywania statusu stacji rowerowych na podstawie
ich aktualnego stanu i warunków środowiskowych.

Klasteryzowane cechy:
- Dane o stacji: 'bikes_available', 'docks_available', 'capacity',
    'manual_bikes_available', 'electric_bikes_available'.
    Cechy te opisują bieżącą dynamikę i pojemność stacji, co jest kluczowe
    dla oceny dostępności rowerów.
- Dane pogodowe: 'temperature', 'wind_kph', 'precip_mm', 'humidity',
    'weather_condition'.
    Warunki pogodowe silnie korelują z popytem na rowery miejskie i ich
    dostępnością na stacjach.
- Dane o jakości powietrza: 'fine_particles_pm2_5', 'coarse_particles_pm10'.
    Jakość powietrza może wpływać na decyzje użytkowników o korzystaniu z rowerów,
    a tym samym na dostępność na stacjach.
- 'cluster_id': Identyfikator klastra, do którego należy dana stacja,
    pochodzący z wcześniej przeprowadzonej klasteryzacji. Dostarcza modelowi
    dodatkowy kontekst o typowych wzorcach zachowań dla tej grupy stacji.

Celem jest przewidywanie dostępności rowerów na stacjach (binarnie:
wystarczająca/niska; wieloklasowo: brak/niska/umiarkowana/wysoka dostępność;
regresja: przewidywana liczba dostępnych rowerów) w oparciu o stan stacji
i czynniki zewnętrzne.
"""

class BikeModelPredictor(BasePredictor):
    
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
            'weather_condition': 'weather_condition',
            'fine_particles_pm2_5': 'fine_particles_pm2_5',
            'coarse_particles_pm10': 'coarse_particles_pm10',
            'cluster_id': 'cluster_id'
        }

    # Wykonuje predykcję na podstawie słownika danych wejściowych.
    # Zwraca przewidywaną wartość/klasę lub (klasę numeryczną, prawdopodobieństwa, klasę tekstową) dla klasyfikacji.
    def predict(self, data_dict: Dict) -> Optional[Union[float, Tuple[int, List[float], str]]]:

        if not self.is_loaded:
            print(f"⚠️ Model '{self.raw_model_name}' nie jest załadowany. Nie można wykonać predykcji.")
            return None
        
        # Przygotuj cechy
        features_df = self._prepare_features(data_dict, self.feature_mapping)
        
        if features_df is None:
            print(f"⚠️ Nie można przygotować cech dla modelu '{self.raw_model_name}'.")
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
            print(f"❌ Błąd predykcji dla modelu '{self.raw_model_name}': {e}")
            return None

# +--------------------------------------------------+
# |          GLOBALNE USTAWIENIA I FUNKCJE           |
# |             Funkcje do zarządzania               |
# +--------------------------------------------------+

# Globalne instancje predyktorów dla łatwego dostępu
# Będą ładowane przy pierwszym imporcie tego pliku
bus_binary_predictor = BusModelPredictor('binary', 'bus_binary_model.pkl')
bus_multiclass_predictor = BusModelPredictor('multiclass', 'bus_multiclass_model.pkl')
bus_regression_predictor = BusModelPredictor('regression', 'bus_regression_model.pkl')

bike_binary_predictor = BikeModelPredictor('binary', 'bike_binary_model.pkl')
bike_multiclass_predictor = BikeModelPredictor('multiclass', 'bike_multiclass_model.pkl')
bike_regression_predictor = BikeModelPredictor('regression', 'bike_regression_model.pkl')

# Przeładowuje wszystkie globalne instancje predyktorów.
def reload_all_predictors() -> Dict:
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

# Zwraca status załadowania dla wszystkich globalnych predyktorów.
def get_all_predictors_status() -> Dict:

    status = {}
    for predictor in [
        bus_binary_predictor, bus_multiclass_predictor, bus_regression_predictor,
        bike_binary_predictor, bike_multiclass_predictor, bike_regression_predictor
    ]:
        if predictor:
            status[predictor.model_name] = predictor.get_status()
    return status
