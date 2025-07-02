from kafka import KafkaConsumer
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych---
from shared.db_utils import save_log, save_bike_cluster_record
from shared.preprocessing_utils import enrich_data_with_environment, rename_keys, replace_nulls
from shared.clusterization.clusterization import BikeStationClusterPredictor
bikes_predictor = BikeStationClusterPredictor()

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





# CHWILOWE
# CHWILOWE
# CHWILOWE
# CHWILOWE

import os

print("🔍 DEBUGOWANIE STRUKTURY PLIKÓW:")
print(f"Bieżący katalog: {os.getcwd()}")
print(f"Zawartość /app/: {os.listdir('/app/')}")
print(f"Zawartość /app/shared/: {os.listdir('/app/shared/')}")
print(f"Zawartość /app/shared/clusterization/: {os.listdir('/app/shared/clusterization/')}")

if os.path.exists('/app/shared/clusterization/models/'):
    print(f"✅ Katalog models istnieje")
    print(f"Zawartość /app/shared/clusterization/models/: {os.listdir('/app/shared/clusterization/models/')}")
    
    model_path = '/app/shared/clusterization/models/bikes_kmeans.pkl'
    print(f"Sprawdzam konkretny plik: {model_path}")
    print(f"Czy istnieje: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        stat_info = os.stat(model_path)
        print(f"Rozmiar pliku: {stat_info.st_size} bajtów")
        print(f"Uprawnienia: {oct(stat_info.st_mode)}")
else:
    print("❌ Katalog models NIE istnieje")

print("🔍 KONIEC DEBUGOWANIA")

# CHWILOWE
# CHWILOWE
# CHWILOWE
# CHWILOWE





print(f"✅ Subskrybent działa na topicu '{KAFKA_TOPIC}'...")
bikes_predictor.load_model()

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

        cluster_id = bikes_predictor.predict_cluster_from_dict(enriched)
        
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




        # (Tu można np. zapisać do SQL, pliku lub dalszego procesu ML)




except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany ręcznie (Ctrl+C).")
    save_log("subscriber_bikes", "info", "Subskrybent zatrzymany ręcznie.")
except Exception as e:
    print(f"❌ Błąd krytyczny subskrybenta: {str(e)}")
    save_log("subscriber_bikes", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
    print("🧹 Połączenie z Kafka zakończone.")
