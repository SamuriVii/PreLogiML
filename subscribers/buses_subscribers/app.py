from kafka import KafkaConsumer
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL ---
from shared.db_utils import save_log
from shared.preprocessing_utils import enrich_data_with_environment, refactor_buses_data, rename_keys, replace_nulls

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

try:
    for message in consumer:
        data = message.value
        data = refactor_buses_data(data)

        print("📨 Odebrano wiadomość:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # Wywołujemy funkcję wzbogadzającą dane buses o dane środowiskowe (environment)
        enriched = enrich_data_with_environment("subscriber_buses", data)
        enriched = rename_keys(enriched)
        enriched = replace_nulls(enriched)

        print("🧠 Wzbogacone dane:")
        print(json.dumps(enriched, indent=2, ensure_ascii=False))

        # (Tu można np. zapisać do SQL, pliku lub dalszego procesu ML)










except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany ręcznie (Ctrl+C).")
    save_log("subscriber_buses", "info", "Subskrybent zatrzymany ręcznie.")
except Exception as e:
    print(f"❌ Błąd krytyczny subskrybenta: {str(e)}")
    save_log("subscriber_buses", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
    print("🧹 Połączenie z Kafka zakończone.")
