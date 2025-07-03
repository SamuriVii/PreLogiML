from typing import Dict, Optional
import pandas as pd
import joblib
import os

class BikeStationClusterPredictor:
    def __init__(self, model_path='/app/shared/clusterization/models/bikes_kmeans.pkl'):
        """Inicjalizuje predyktor z wczytanym modelem"""
        self.model_path = model_path
        self.model_data = None
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        self.load_model()
    
    def load_model(self):
        """Wczytuje zapisany model i wszystkie komponenty"""
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
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            self.is_loaded = False
            return False

    def prepare_features_from_dict(self, enriched_dict: Dict) -> Optional[pd.DataFrame]:
        """Przygotowuje cechy z pojedynczego dict'a (dla real-time predykcji)"""
        
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
            
            return df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech: {e}")
            return None
    
    def predict_cluster_from_dict(self, enriched_dict: Dict) -> Optional[int]:
        """Przewiduje klaster dla pojedynczego dict'a danych"""
        
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
            
            return int(cluster_prediction[0])
            
        except Exception as e:
            print(f"❌ Błąd predykcji klastra: {e}")
            return None
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """Zwraca informacje o centrum klastra"""
        if not self.is_loaded or cluster_id >= len(self.kmeans.cluster_centers_):
            return None
        
        center = self.kmeans.cluster_centers_[cluster_id]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(center))]
        
        cluster_info = {}
        for feature, value in zip(feature_names, center):
            cluster_info[feature] = float(value)
        
        return cluster_info
    
    def reload_model(self) -> bool:
        """Przeładowuje model z dysku"""
        print("🔄 Przeładowywanie modelu...")
        return self.load_model()

class BusClusterPredictor:
    def __init__(self, model_path='/app/shared/clusterization/models/buses_kmeans.pkl'):
        """Inicjalizuje predyktor z wczytanym modelem"""
        self.model_path = model_path
        self.model_data = None
        self.kmeans = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        self.load_model()
    
    def load_model(self):
        """Wczytuje zapisany model i wszystkie komponenty"""
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
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            self.is_loaded = False
            return False

    def prepare_features_from_dict(self, enriched_dict: Dict) -> Optional[pd.DataFrame]:
        """Przygotowuje cechy z pojedynczego dict'a (dla real-time predykcji)"""
        
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
            
            return df
            
        except Exception as e:
            print(f"❌ Błąd przygotowywania cech: {e}")
            return None
    
    def predict_cluster_from_dict(self, enriched_dict: Dict) -> Optional[int]:
        """Przewiduje klaster dla pojedynczego dict'a danych"""
        
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
            
            return int(cluster_prediction[0])
            
        except Exception as e:
            print(f"❌ Błąd predykcji klastra: {e}")
            return None
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """Zwraca informacje o centrum klastra"""
        if not self.is_loaded or cluster_id >= len(self.kmeans.cluster_centers_):
            return None
        
        center = self.kmeans.cluster_centers_[cluster_id]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(center))]
        
        cluster_info = {}
        for feature, value in zip(feature_names, center):
            cluster_info[feature] = float(value)
        
        return cluster_info
    
    def reload_model(self) -> bool:
        """Przeładowuje model z dysku"""
        print("🔄 Przeładowywanie modelu...")
        return self.load_model()

# Dodatkowe funkcje
def reload_models():
    """Przeładowuje wszystkie modele z dysku"""
    global _bike_predictor, _bus_predictor
    
    print("🔄 Przeładowywanie modeli...")
    
    if _bike_predictor:
        _bike_predictor.reload_model()
    
    if _bus_predictor:
        _bus_predictor.reload_model()
        pass
    
    print("✅ Przeładowywanie modeli zakończone")

def get_models_status() -> Dict:
    """Zwraca status wszystkich modeli"""
    global _bike_predictor, _bus_predictor
    
    return {
        'bike_predictor': {
            'loaded': _bike_predictor is not None and _bike_predictor.is_loaded,
            'model_path': _bike_predictor.model_path if _bike_predictor else None,
            'n_clusters': _bike_predictor.model_data.get('n_clusters') if _bike_predictor and _bike_predictor.is_loaded else None
        },
        'bus_predictor': {
            'loaded': _bus_predictor is not None and getattr(_bus_predictor, 'is_loaded', False),
            'model_path': _bus_predictor.model_path if _bus_predictor else None
        }
    }
