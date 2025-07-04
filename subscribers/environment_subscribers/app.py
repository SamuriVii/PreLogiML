from datetime import datetime, timezone
from kafka import KafkaConsumer
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_dto import EnvironmentEntry
from shared.db_conn import SessionLocal
from shared.db_utils import save_log

# --- Ustawienia podstawowe ---
KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "environment"
KAFKA_GROUP = "environment-subscriber"  

# --- Ustawienie Kafka Subscriber ---
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id=KAFKA_GROUP,
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    fetch_min_bytes=1,
    fetch_max_wait_ms=500,
    max_poll_records=1,
    session_timeout_ms=30000,
    heartbeat_interval_ms=3000
)

# +-------------------------------------+
# |       GŁÓWNA CZĘŚC WYKONUJĄCA       |
# |      Proces przetwarzania danych    |
# +-------------------------------------+

print(f"✅ Subskrybent działa na topicu '{KAFKA_TOPIC}'...")
save_log("subscriber_environment", "info", "Subscriber uruchomiony.")

try:
    for message in consumer:
        data = message.value
        save_log("subscriber_environment", "info", "Odebrano dane środowiskowe z Kafki.")
        try:
            db = SessionLocal()

            entry = EnvironmentEntry(
                type=data.get("type", "environment"),
                timestamp=datetime.fromtimestamp(data.get("timestamp", time.time()), tz=timezone.utc),
                city=data.get("location", {}).get("city"),
                lat=data.get("location", {}).get("lat"),
                lon=data.get("location", {}).get("lon"),
                temperature=data.get("weather", {}).get("temperature"),
                feelslike=data.get("weather", {}).get("feelslike"),
                humidity=data.get("weather", {}).get("humidity"),
                wind_kph=data.get("weather", {}).get("wind_kph"),
                precip_mm=data.get("weather", {}).get("precip_mm"),
                cloud=data.get("weather", {}).get("cloud"),
                visibility_km=data.get("weather", {}).get("visibility_km"),
                uv_index=data.get("weather", {}).get("uv_index"),
                is_day=data.get("weather", {}).get("is_day"),
                condition=data.get("weather", {}).get("condition"),
                pm2_5=data.get("air_quality", {}).get("pm2_5"),
                pm10=data.get("air_quality", {}).get("pm10"),
                co=data.get("air_quality", {}).get("co"),
                no2=data.get("air_quality", {}).get("no2"),
                o3=data.get("air_quality", {}).get("o3"),
                so2=data.get("air_quality", {}).get("so2")
            )
            db.add(entry)
            db.commit()
            db.close()

            save_log("subscriber_environment", "info", "Zapisano dane środowiskowe do bazy.")
        except Exception as e:
                print(f"❌ Błąd zapisu do bazy: {e}")
                save_log("subscriber_environment", "error", f"Błąd zapisu danych: {str(e)}")
except KeyboardInterrupt:
    print("⛔ Subskrybent zatrzymany.")
    save_log("subscriber_environment", "info", "Subskrybent zatrzymany.")
except Exception as e:
    save_log("subscriber_environment", "error", f"Błąd subskrybenta: {str(e)}")
finally:
    consumer.close()
