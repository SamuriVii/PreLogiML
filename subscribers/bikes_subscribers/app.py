from typing import Optional, List, Tuple
from kafka import KafkaConsumer
import json
import time

import random
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych---
from shared.db_utils import save_log, save_bike_cluster_record, save_bike_class_record
from shared.preprocessing_utils import enrich_data_with_environment, rename_keys, replace_nulls, create_bike_summary_sentence
from shared.clusterization.clusterization import BikeStationClusterPredictor
from shared.classification.classification import bike_binary_predictor, bike_multiclass_predictor, bike_regression_predictor, get_all_predictors_status
bikes_cluster_predictor = BikeStationClusterPredictor()

# --- Ustawienia podstawowe ---
KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "bikes"
KAFKA_GROUP = "bikes-subscriber"  

# --- Ustawienie Kafka Subscriber ---
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id=KAFKA_GROUP,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print(f"✅ Subskrybent działa na topicu '{KAFKA_TOPIC}'...")
bikes_cluster_predictor.load_model()

print("\n--- Status załadowanych modeli klasyfikacji/regresji dla rowerów ---")
current_model_statuses = get_all_predictors_status()
# Filtrujemy tylko te, które dotyczą rowerów dla czytelności logów
bike_model_statuses = {k: v for k, v in current_model_statuses.items() if v['data_source'] == 'bike'}

for model_name, status_info in bike_model_statuses.items():
    print(f"  Model: {model_name}, Załadowany: {status_info['loaded']}, Ścieżka: {status_info['model_path']}")
    if not status_info['loaded']:
        print(f"    Wiadomość statusu: {status_info['status_message']}")

# Model zostanie pobrany do lokalnego cache'u przy pierwszym uruchomieniu
print("🔄 Ładowanie modelu SentenceTransformer: all-MiniLM-L6-v2...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model SentenceTransformer załadowany pomyślnie!")
except Exception as e:
    embedding_model = None
    print(f"❌ Błąd ładowania modelu SentenceTransformer: {e}")
    print("   Upewnij się, że masz połączenie z internetem (przy pierwszym uruchomieniu) i biblioteka jest zainstalowana.")

# --- Funkcja do generowania embeddingu (teraz używa SentenceTransformer) ---
def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generuje embedding dla danego tekstu za pomocą załadowanego modelu.
    """
    if embedding_model is None:
        print("⚠️ Model embeddingowy nie załadowany. Nie można wygenerować embeddingu.")
        return None
    try:
        # Kodowanie tekstu i konwersja na listę Pythona
        embedding = embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"❌ Błąd podczas generowania embeddingu dla tekstu '{text[:50]}...': {e}")
        return None

try:
    for message in consumer:
        data = message.value
        vehicle = data.pop("vehicle_types_available", {})
        data["bike_available"] = vehicle.get("bike", 0)
        data["ebike_available"] = vehicle.get("ebike", 0)

        print("🧠 Odebrano wiadomość:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # Wywołujemy funkcję wzbogadzającą dane bikes o dane środowiskowe (environment)
        enriched = enrich_data_with_environment("subscriber_bikes", data)
        enriched = rename_keys(enriched)
        enriched = replace_nulls(enriched)

        save_bike_cluster_record(enriched)

        print("🧠🧠🧠 Wzbogacone dane:")
        print(json.dumps(enriched, indent=2, ensure_ascii=False))

        cluster_id = bikes_cluster_predictor.predict_cluster_from_dict(enriched)
        
        if cluster_id is not None:
            enriched['cluster_id'] = cluster_id
            enriched['cluster_prediction_success'] = True
            print(f"🎯 Przewidziano klaster: {cluster_id}")
        else:
            enriched['cluster_id'] = None
            enriched['cluster_prediction_success'] = False
            print("⚠️ Nie udało się przewidzieć klastra")

        save_bike_class_record(enriched)

        print("🧠🧠🧠🧠🧠🧠 Wzbogacone dane (z klastrem):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # Predykcja binarna
        if bike_binary_predictor.is_loaded:
            binary_pred_result = bike_binary_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest krotką o 3 elementach
            if isinstance(binary_pred_result, Tuple) and len(binary_pred_result) == 3:
                prediction_num, probabilities, prediction_label = binary_pred_result  # Prawidłowe rozpakowanie
                enriched['bike_binary_prediction'] = prediction_num
                enriched['bike_binary_probabilities'] = probabilities
                enriched['bike_binary_prediction_success'] = True
                enriched['bike_binary_label'] = prediction_label  # Zapisujemy etykietę tekstową
                print(f"🎯 Predykcja binarna dla rowerów: {prediction_num} ({enriched['bike_binary_label']}), Prawdopodobieństwa: {probabilities}")
            else:
                enriched['bike_binary_prediction'] = None
                enriched['bike_binary_probabilities'] = None
                enriched['bike_binary_prediction_success'] = False
                enriched['bike_binary_label'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji binarnej dla rowerów: {binary_pred_result}")
        else:
            enriched['bike_binary_prediction'] = None
            enriched['bike_binary_probabilities'] = None
            enriched['bike_binary_prediction_success'] = False
            enriched['bike_binary_label'] = None
            print(f"⚠️ Model {bike_binary_predictor.model_name} nie załadowany, pomijam predykcję binarną dla rowerów.")

        # Predykcja wieloklasowa
        if bike_multiclass_predictor.is_loaded:
            multiclass_pred_result = bike_multiclass_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest krotką o 3 elementach
            if isinstance(multiclass_pred_result, Tuple) and len(multiclass_pred_result) == 3:
                prediction_num, probabilities, prediction_label = multiclass_pred_result  # Prawidłowe rozpakowanie
                enriched['bike_multiclass_prediction'] = prediction_num
                enriched['bike_multiclass_probabilities'] = probabilities
                enriched['bike_multiclass_prediction_success'] = True
                enriched['bike_multiclass_label'] = prediction_label  # Zapisujemy etykietę tekstową
                print(f"🎯 Predykcja wieloklasowa dla rowerów: {prediction_num} ({enriched['bike_multiclass_label']}), Prawdopodobieństwa: {probabilities}")
            else:
                enriched['bike_multiclass_prediction'] = None
                enriched['bike_multiclass_probabilities'] = None
                enriched['bike_multiclass_prediction_success'] = False
                enriched['bike_multiclass_label'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji wieloklasowej dla rowerów: {multiclass_pred_result}")
        else:
            enriched['bike_multiclass_prediction'] = None
            enriched['bike_multiclass_probabilities'] = None
            enriched['bike_multiclass_prediction_success'] = False
            enriched['bike_multiclass_label'] = None
            print(f"⚠️ Model {bike_multiclass_predictor.model_name} nie załadowany, pomijam predykcję wieloklasową dla rowerów.")

        # Predykcja regresji
        if bike_regression_predictor.is_loaded:
            regression_prediction = bike_regression_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest liczbą zmiennoprzecinkową
            if isinstance(regression_prediction, float):
                # Zapisujemy oryginalną predykcję przed obcięciem
                enriched['bike_regression_prediction_original'] = regression_prediction
                # Obcięcie do 0, jeśli jest ujemne
                clipped_prediction = max(0, regression_prediction)
                enriched['bike_regression_prediction'] = clipped_prediction
                enriched['bike_regression_prediction_success'] = True
                print(f"🎯 Predykcja regresji dla rowerów: {clipped_prediction:.2f} (oryginalna: {regression_prediction:.2f})")
                if regression_prediction < 0:
                    print("⚠️ UWAGA: Przewidywana liczba rowerów była ujemna i została obcięta do 0.")
            else:
                enriched['bike_regression_prediction'] = None
                enriched['bike_regression_prediction_success'] = False
                enriched['bike_regression_prediction_original'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji regresji dla rowerów: {regression_prediction}")
        else:
            enriched['bike_regression_prediction'] = None
            enriched['bike_regression_prediction_success'] = False
            enriched['bike_regression_prediction_original'] = None
            print(f"⚠️ Model {bike_regression_predictor.model_name} nie załadowany, pomijam predykcję regresji dla rowerów.")

        # save_bike_class_record(enriched) # Ta linia została usunięta, aby nie zapisywać danych do starej tabeli

        print("🔥🔥🔥🔥🔥🔥 Wzbogacone dane (z klastrem i predykcjami klasyfikacji/regresji):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # --- Tworzenie zdania podsumowującego (nowa część) ---
        summary_sentence = create_bike_summary_sentence(enriched)
        print(f"\n📝 Wygenerowane zdanie podsumowujące: {summary_sentence}")

        # --- Następny krok: generowanie embeddingu (tylko print, bez faktycznego generowania na razie) ---
        print("\n➡️ Gotowy do generowania embeddingu za pomocą SentenceTransformers: all-MiniLM-L6-v2.")
        print("   (W tym miejscu wywołałbyś model embeddingowy dla zdania: '{summary_sentence}')")

        # --- Generowanie i zapis embeddingu ---
        embedding = generate_embedding(summary_sentence)
        if embedding is not None:
            print("\n➡️ Wygenerowany Embedding:")
            print(embedding)
        else:
            save_log("subscriber_bikes", "error", "Nie udało się wygenerować embeddingu dla roweru.")






except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany ręcznie (Ctrl+C).")
    save_log("subscriber_bikes", "info", "Subskrybent zatrzymany ręcznie.")
except Exception as e:
    print(f"❌ Błąd krytyczny subskrybenta: {str(e)}")
    save_log("subscriber_bikes", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
    print("🧹 Połączenie z Kafka zakończone.")
