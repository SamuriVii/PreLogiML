from sentence_transformers import SentenceTransformer
from typing import Optional, List, Tuple
from kafka import KafkaConsumer
import numpy as np
import random
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(180)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_utils import save_log, save_bus_data_to_base
from shared.preprocessing_utils import enrich_data_with_environment, refactor_buses_data, rename_keys, replace_nulls, create_bus_summary_sentence, prepare_sql_record_all_fields, prepare_vector_db_record_all_fields
from shared.clusterization.clusterization import bus_cluster_predictor
from shared.classification.classification import bus_binary_predictor, bus_multiclass_predictor, bus_regression_predictor, get_all_predictors_status 

# --- Ustawienia podstawowe ---
KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "buses"
KAFKA_GROUP = "buses-subscriber"  

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

# +-------------------------------------+
# |      CZĘŚĆ ŁADUJĄCA MODEL ST        |
# |     Proces przetwarzania danych     |
# +-------------------------------------+

# Model zostanie pobrany do lokalnego cache'u przy pierwszym uruchomieniu
print("🔄 Ładowanie modelu SentenceTransformer: all-MiniLM-L6-v2...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model SentenceTransformer załadowany pomyślnie!")
    save_log("buses_subscriber", "info", "Model embeddingowy załadowany pomyślnie.")
except Exception as e:
    embedding_model = None
    print(f"❌ Błąd ładowania modelu SentenceTransformer: {e}")
    save_log("buses_subscriber", "error", f"Błąd w trakcie ładowania modelu embeddingowego: {e}")

# --- Funkcja do generowania embeddingu (teraz używa SentenceTransformer) ---
def generate_embedding(text: str) -> Optional[List[float]]:
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
        data = refactor_buses_data(data)

        print("🧠 🧠 Odebrano wiadomość:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # Wywołujemy funkcję wzbogadzającą dane buses o dane środowiskowe (environment)
        enriched = enrich_data_with_environment("subscriber_buses", data)
        enriched = rename_keys(enriched)
        enriched = replace_nulls(enriched)

        if any(value is None for value in enriched.values()):
            print("⚠️ Wykryto wartości None w danych po przetworzeniu. Pomijam dalsze przetwarzanie i przechodzę do następnej wiadomości.")
            save_log("subscriber_buses", "warning", "Wykryto wartości None w danych po preprocessing. Pominięto wiadomość.")
            continue

        print("🧠 🧠 🧠 🧠 Wzbogacone dane:")
        print(json.dumps(enriched, indent=2, ensure_ascii=False))

        # +-------------------------------------+
        # |         CZĘŚĆ KLASTROWANIA          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

        cluster_id = bus_cluster_predictor.predict_cluster_from_dict(enriched)
        
        if cluster_id is not None:
            enriched['cluster_id'] = cluster_id
            enriched['cluster_prediction_success'] = True
            print(f"🎯 Przewidziano klaster: {cluster_id}")
        else:
            enriched['cluster_id'] = None
            enriched['cluster_prediction_success'] = False
            print("⚠️ Nie udało się przewidzieć klastra")
        
        print("🧠 🧠 🧠 🧠 🧠 🧠 🧠 🧠 Wzbogacone dane (z klastrem):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # +-------------------------------------+
        # |         CZĘŚĆ KLASYFIKACJI          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

        # Predykcja binarna
        if bus_binary_predictor.is_loaded:
            binary_pred_result = bus_binary_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest krotką o 3 elementach
            if isinstance(binary_pred_result, Tuple) and len(binary_pred_result) == 3:
                prediction_num, probabilities, prediction_label = binary_pred_result # Prawidłowe rozpakowanie
                enriched['is_late_prediction'] = prediction_num
                enriched['is_late_probabilities'] = probabilities
                enriched['is_late_prediction_success'] = True
                enriched['is_late_label'] = prediction_label # Zapisujemy etykietę tekstową
                print(f"🎯 Predykcja binarna (is_late): {prediction_num} ({enriched['is_late_label']}), Prawdopodobieństwa: {probabilities}")
            else:
                enriched['is_late_prediction'] = None
                enriched['is_late_probabilities'] = None
                enriched['is_late_prediction_success'] = False
                enriched['is_late_label'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji binarnej dla autobusu: {binary_pred_result}")
        else:
            enriched['is_late_prediction'] = None
            enriched['is_late_probabilities'] = None
            enriched['is_late_prediction_success'] = False
            enriched['is_late_label'] = None
            print(f"⚠️ Model {bus_binary_predictor.model_name} nie załadowany, pomijam predykcję binarną.")

        # Predykcja wieloklasowa (delay_category)
        if bus_multiclass_predictor.is_loaded:
            multiclass_pred_result = bus_multiclass_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest krotką o 3 elementach
            if isinstance(multiclass_pred_result, Tuple) and len(multiclass_pred_result) == 3:
                prediction_num, probabilities, prediction_label = multiclass_pred_result # Prawidłowe rozpakowanie
                enriched['delay_category_prediction'] = prediction_num
                enriched['delay_category_probabilities'] = probabilities
                enriched['delay_category_prediction_success'] = True
                enriched['delay_category_label'] = prediction_label # Zapisujemy etykietę tekstową
                print(f"🎯 Predykcja wieloklasowa (delay_category): {prediction_num} ({enriched['delay_category_label']}), Prawdopodobieństwa: {probabilities}")
            else:
                enriched['delay_category_prediction'] = None
                enriched['delay_category_probabilities'] = None
                enriched['delay_category_prediction_success'] = False
                enriched['delay_category_label'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji wieloklasowej dla autobusu: {multiclass_pred_result}")
        else:
            enriched['delay_category_prediction'] = None
            enriched['delay_category_probabilities'] = None
            enriched['delay_category_prediction_success'] = False
            enriched['delay_category_label'] = None
            print(f"⚠️ Model {bus_multiclass_predictor.model_name} nie załadowany, pomijam predykcję wieloklasową.")

        # Predykcja regresji (average_delay_seconds)
        if bus_regression_predictor.is_loaded:
            regression_prediction = bus_regression_predictor.predict(enriched)
            # Sprawdzamy, czy wynik jest liczbą zmiennoprzecinkową
            if isinstance(regression_prediction, float):
                # Zapisujemy oryginalną predykcję przed obcięciem, do celów logowania/debugowania
                enriched['average_delay_seconds_prediction_original'] = regression_prediction
                # Obcięcie wartości do minimum 0
                clipped_prediction = max(0, regression_prediction) 
                
                enriched['average_delay_seconds_prediction'] = clipped_prediction
                enriched['average_delay_seconds_prediction_success'] = True
                print(f"🎯 Predykcja regresji (average_delay_seconds): {clipped_prediction:.2f} sekund (oryginalna: {regression_prediction:.2f})")
                if regression_prediction < 0:
                    print("⚠️ UWAGA: Przewidywane opóźnienie było ujemne i zostało obcięte do 0.")
            else:
                enriched['average_delay_seconds_prediction'] = None
                enriched['average_delay_seconds_prediction_success'] = False
                enriched['average_delay_seconds_prediction_original'] = None
                print(f"⚠️ Błąd: Nieprawidłowy wynik predykcji regresji dla autobusu: {regression_prediction}")
        else:
            enriched['average_delay_seconds_prediction'] = None
            enriched['average_delay_seconds_prediction_success'] = False
            enriched['average_delay_seconds_prediction_original'] = None
            print(f"⚠️ Model {bus_regression_predictor.model_name} nie załadowany, pomijam predykcję regresji.")

        print("🔥🔥🔥🔥🔥🔥 Wzbogacone dane (z klastrem i predykcjami klasyfikacji/regresji):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # +-------------------------------------+
        # |         CZĘŚĆ EMBEDDINGOWA          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

        # --- Tworzenie zdania podsumowującego ---
        summary_sentence = create_bus_summary_sentence(enriched)
        print(f"\n📝 Wygenerowane zdanie podsumowujące: {summary_sentence}")

        # --- Generowanie i zapis embeddingu ---
        embedding = generate_embedding(summary_sentence)
        if embedding is not None:
            print("\n➡️ Wygenerowany Embedding:")
            print(embedding)
            save_log("subscriber_buses", "info", "Wygenerowano embedding dla danych autobusowych.")
        else:
            save_log("subscriber_buses", "error", "Nie udało się wygenerować embeddingu dla danych autobusowych.")

        # +----------------------------------------+
        # |  ŁĄCZENIE DANYCH I WYSYŁANIE DO BAZY   |
        # |     Proces przetwarzania danych        |
        # +----------------------------------------+

        # Krok 1: Przygotowanie danych dla bazy SQL
        final_sql_data = prepare_sql_record_all_fields(enriched, summary_sentence)
        print("\n📊 Dane przygotowane dla bazy SQL (wszystkie pola zachowane):")
        print(json.dumps(final_sql_data, indent=2, ensure_ascii=False, default=str))

        # Krok 2: Przygotowanie danych dla bazy wektorowej
        final_vector_db_data = prepare_vector_db_record_all_fields(enriched, summary_sentence, embedding)
        if final_vector_db_data:
            print("\n🗃️ Struktura przygotowana dla bazy wektorowej (wszystkie metadane zachowane):")
            # Dla czytelności, nie drukujemy całego wektora, tylko jego początek
            printable_vector_db_data = final_vector_db_data.copy()
            if printable_vector_db_data['vector']:
                printable_vector_db_data['vector_preview'] = printable_vector_db_data['vector'][:5]
                del printable_vector_db_data['vector'] # Usuwamy pełny wektor, żeby logi nie były zbyt długie
            print(json.dumps(printable_vector_db_data, indent=2, ensure_ascii=False, default=str))
            save_log("subscriber_buses", "info", "Dane przygotowane dla bazy wektorowej.")
        else:
            save_log("subscriber_buses", "error", "Nie udało się przygotować danych dla bazy wektorowej.")

        # Krok 3: Zapisanie danych do SQL oraz wysłanie danych do bazy wektorowej
        save_bus_data_to_base(final_sql_data)





except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany ręcznie (Ctrl+C).")
    save_log("subscriber_buses", "info", "Subskrybent zatrzymany ręcznie.")
except Exception as e:
    print(f"❌ Błąd krytyczny subskrybenta: {str(e)}")
    save_log("subscriber_buses", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
    print("🧹 Połączenie z Kafka zakończone.")
