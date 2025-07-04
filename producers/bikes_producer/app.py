from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer
from typing import List, Dict
import requests
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(180)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL ---
from shared.db_utils import save_log

# --- Ustawienia podstawowe ---
STATION_INFO_URL = 'https://gbfs.urbansharing.com/rowermevo.pl/station_information.json'
STATION_STATUS_URL = 'https://gbfs.urbansharing.com/rowermevo.pl/station_status.json'

KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "bikes"

# --- Ustawienie Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# --- Funkcje Pomocnicze ---
def get_timestamp():
    return int(datetime.now(timezone.utc).timestamp())

def filter_stations_by_prefix(stations: List[Dict], prefix: str) -> List[Dict]:
    return [s for s in stations if s.get("name", "").startswith(prefix)]

def extract_vehicle_types(raw_vehicle_types: List[Dict]) -> Dict[str, int]:
    return {
        vt.get("vehicle_type_id"): vt.get("count", 0)
        for vt in raw_vehicle_types
        if vt.get("vehicle_type_id")
    }

def process_mevo_stations(info_data: List[Dict], status_data: List[Dict]) -> List[Dict]:
    status_dict = {s["station_id"]: s for s in status_data}
    simplified_stations = []

    for station in info_data:
        station_id = station.get("station_id")
        status = status_dict.get(station_id, {})

        simplified = {
            "timestamp": get_timestamp(),
            "station_id": station_id,
            "name": station.get("name", "Brak nazwy"),
            "capacity": station.get("capacity", 0),
            "bikes_available": status.get("num_bikes_available", 0),
            "docks_available": status.get("num_docks_available", 0),
            "vehicle_types_available": extract_vehicle_types(status.get("vehicle_types_available", []))
        }

        simplified_stations.append(simplified)

    gda_stations = filter_stations_by_prefix(simplified_stations, "GDA")
    gda_stations.sort(key=lambda s: s.get("name", ""))

    return gda_stations

# --- Funkcja pobierająca dane z Mevo ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=60, min=60, max=300))
def fetch_mevo_data() -> List[Dict]:
    info_resp = requests.get(STATION_INFO_URL, timeout=10)
    status_resp = requests.get(STATION_STATUS_URL, timeout=10)

    if info_resp.status_code == 200 and status_resp.status_code == 200:
        info_data = info_resp.json().get("data", {}).get("stations", [])
        status_data = status_resp.json().get("data", {}).get("stations", [])

        return process_mevo_stations(info_data, status_data)
    else:
        raise Exception(f"Błąd API Mevo: info={info_resp.status_code}, status={status_resp.status_code}")

# --- Funkcja wysyłająca dane do Kafki ---
def send_to_kafka(payload: dict):
    try:
        producer.send(KAFKA_TOPIC, payload)
        producer.flush()
    except Exception as e:
        msg = (f"Błąd wysyłania do Kafki: {str(e)}")
        save_log("producer_bikes", "error", msg)

# --- Funkcja wykonująca funkcję pobierania danych z Mevo oraz następnie wysyłająca dane do Kafki ---
def job():
    try:
        stations_payload = fetch_mevo_data()
        if stations_payload:
            success_count = 0
            for station in stations_payload:
                try:
                    send_to_kafka(station)
                    success_count += 1
                except Exception:
                    # Błąd już zalogowany w send_to_kafka
                    pass
            
            # Loguj po całym batchu
            save_log("producer_bikes", "info", f"Wysłano {success_count}/{len(stations_payload)} stacji do Kafki.")
        else:
            save_log("producer_bikes", "warning", "Pobrano pustą odpowiedź z API Mevo.")
    except RetryError:
        msg = "Nie udało się pobrać danych po 5 próbach."
        save_log("producer_bikes", "error", msg)

# --- Główna część ---
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(job, 'interval', minutes=1)
    scheduler.start()

    save_log("producer_bikes", "info", "Producer uruchomiony.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        producer.close()
        print("⛔ Bikes Producer zatrzymany.")
        save_log("producer_bikes", "info", "Producer zatrzymany.")
    except Exception as e:
        save_log("producer_bikes", "error", f"Błąd krytyczny: {str(e)}")
    finally:
        scheduler.shutdown()
        producer.close()
