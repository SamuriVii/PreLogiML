from typing import Optional, List, Tuple
from kafka import KafkaConsumer
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(180)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_utils import save_log, save_bike_data_to_base
from shared.preprocessing_utils import enrich_data_with_environment, rename_keys, replace_nulls, create_bike_summary_sentence, prepare_sql_record_all_fields, prepare_vector_db_record_all_bike_fields
from shared.clusterization.clusterization import bike_cluster_predictor
from shared.classification.classification import bike_binary_predictor, bike_multiclass_predictor, bike_regression_predictor
from shared.anything_wrapper import add_raw_bike_text_to_anythingllm

# --- Ustawienia podstawowe ---
KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "bikes"
KAFKA_GROUP = "bikes-subscriber"  
ANYTHINGLLM_WORKSPACE_SLUG = "project"

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
# |       GŁÓWNA CZĘŚĆ WYKONUJĄCA       |
# |      Proces przetwarzania danych    |
# +-------------------------------------+

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

        if any(value is None for value in enriched.values()):
            print("⚠️ Wykryto wartości None w danych po przetworzeniu. Pomijam dalsze przetwarzanie i przechodzę do następnej wiadomości.")
            save_log("subscriber_bikes", "warning", "Wykryto wartości None w danych po preprocessing. Pominięto wiadomość.")
            continue

        print("🧠🧠🧠 Wzbogacone dane:")
        print(json.dumps(enriched, indent=2, ensure_ascii=False))

        # +-------------------------------------+
        # |         CZĘŚĆ KLASTROWANIA          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

        cluster_id = bike_cluster_predictor.predict_cluster_from_dict(enriched)
        
        if cluster_id is not None:
            enriched['cluster_id'] = cluster_id
            enriched['cluster_prediction_success'] = True
            print(f"🎯 Przewidziano klaster: {cluster_id}")
        else:
            enriched['cluster_id'] = None
            enriched['cluster_prediction_success'] = False
            print("⚠️ Nie udało się przewidzieć klastra")

        print("🧠🧠🧠🧠🧠🧠 Wzbogacone dane (z klastrem):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # +-------------------------------------+
        # |         CZĘŚĆ KLASYFIKACJI          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

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

        print("🔥🔥🔥🔥🔥🔥 Wzbogacone dane (z klastrem i predykcjami klasyfikacji/regresji):")
        print(json.dumps(enriched, indent=2, ensure_ascii=False, default=str))

        # +-------------------------------------+
        # |         CZĘŚĆ EMBEDDINGOWA          |
        # |     Proces przetwarzania danych     |
        # +-------------------------------------+

        # --- Tworzenie zdania podsumowującego ---
        summary_sentence = create_bike_summary_sentence(enriched)
        print(f"\n📝 Wygenerowane zdanie podsumowujące: {summary_sentence}")

        # +----------------------------------------+
        # |  ŁĄCZENIE DANYCH I WYSYŁANIE DO BAZY   |
        # |     Proces przetwarzania danych        |
        # +----------------------------------------+

        # Krok 1: Przygotowanie i wysłanie danych dla bazy SQL
        final_data = prepare_sql_record_all_fields(enriched, summary_sentence)
        print("\n📊 Dane przygotowane dla bazy SQL (wszystkie pola zachowane):")
        print(json.dumps(final_data, indent=2, ensure_ascii=False, default=str))

        # Krok 2: Przygotowanie danych dla bazy wektorowej
        data = prepare_vector_db_record_all_bike_fields(enriched)

        if summary_sentence:
            print("\n🗃️ Struktura przygotowana dla AnythingLLM (tekst i metadane):")
            printable_anythingllm_payload = data.copy()
            print(f"textContent_preview='{summary_sentence[:100]}...', metadata keys={list(printable_anythingllm_payload.keys())}")
            print(f"Pełne metadane: {json.dumps(printable_anythingllm_payload, indent=2, ensure_ascii=False, default=str)}")
            
            save_log("subscriber_bikes", "info", "Dane przygotowane dla AnythingLLM.") # Zmieniono log

            print("\n🚀 Wysyłanie danych do AnythingLLM...")
            add_response = add_raw_bike_text_to_anythingllm(
                ANYTHINGLLM_WORKSPACE_SLUG,
                summary_sentence,
                data
            )
            print("Odpowiedź z AnythingLLM:", add_response)
            if add_response.get("success"):
                save_bike_data_to_base(final_data)
                save_log("subscriber_bikes", "info", f"Tekst dodany do AnythingLLM dla workspace'u {ANYTHINGLLM_WORKSPACE_SLUG}.") # Zmieniono log
            else:
                save_log("subscriber_bikes", "error", f"Błąd podczas dodawania tekstu do AnythingLLM: {add_response.get('message')}")
        else:
            save_log("subscriber_bikes", "error", "Brak summary_sentence, nie wysłano danych do AnythingLLM.")

except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany ręcznie (Ctrl+C).")
    save_log("subscriber_bikes", "info", "Subskrybent zatrzymany ręcznie.")
except Exception as e:
    print(f"❌ Błąd krytyczny subskrybenta: {str(e)}")
    save_log("subscriber_bikes", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
    print("🧹 Połączenie z Kafka zakończone.")
