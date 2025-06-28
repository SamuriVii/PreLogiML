from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer
import requests
import json
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL ---
from shared.db_utils import save_log

# --- Ustawienia podstawowe ---
GPS_URL = "https://ckan2.multimediagdansk.pl/gpsPositions?v=2"
DEPARTURES_URL = "https://ckan2.multimediagdansk.pl/departures"

KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "buses"

# --- Ustawienie Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# --- Funkcje Pomocnicze ---
def get_timestamp():
    return int(datetime.now(timezone.utc).timestamp())

def group_departures_by_vehicle_and_trip(departures_data, gps_data):
    gps_by_vehicle = {v["vehicleId"]: v for v in gps_data.get("vehicles", [])}
    grouped = {}

    for stop_id, stop in departures_data.items():
        for d in stop.get("departures", []):
            vehicle_id = d.get("vehicleId")
            trip_id = d.get("tripId")
            route_id = d.get("routeId")
            if not vehicle_id or trip_id is None:
                continue

            key = (vehicle_id, trip_id)
            gps = gps_by_vehicle.get(vehicle_id, {})

            if key not in grouped:
                grouped[key] = {
                    "timestamp": get_timestamp(),
                    "vehicle_id": vehicle_id,
                    "line": gps.get("routeShortName"),
                    "headsign": gps.get("headsign") or d.get("headsign"),
                    "lat": gps.get("lat"),
                    "lon": gps.get("lon"),
                    "speed": gps.get("speed"),
                    "direction": gps.get("direction"),
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "estimated_stops": []
                }

            grouped[key]["estimated_stops"].append({
                "stop_id": stop_id,
                "delay": d.get("delayInSeconds"),
                "estimated_departure": d.get("estimatedTime"),
                "scheduled_departure": d.get("theoreticalTime")
            })

    return list(grouped.values())

# --- Funkcja pobierająca dane z Tristara ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=60, min=60, max=300))
def fetch_json_data(url):
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise Exception(f"Błąd pobierania z {url}: {resp.status_code}")

# --- Funkcja wysyłająca dane do Kafki ---
def send_to_kafka(payload: dict):
    try:
        producer.send(KAFKA_TOPIC, payload)
        producer.flush()
    except Exception as e:
        msg = (f"Błąd wysyłania do Kafki: {str(e)}")
        save_log("producer_buses", "error", msg)

# --- Funkcja wykonująca funkcję pobierania danych z Tristara oraz następnie wysyłająca dane do Kafki ---
def job():
    try:
        gps_data = fetch_json_data(GPS_URL)
        departures_data = fetch_json_data(DEPARTURES_URL)

        if gps_data and departures_data:
            combined_payload = group_departures_by_vehicle_and_trip(departures_data, gps_data)
            success_count = 0

            for event in combined_payload:
                try:
                    send_to_kafka(event)
                    success_count += 1
                except Exception:
                    pass  # logowanie już jest w send_to_kafka

            save_log("producer_buses", "info", f"Wysłano {success_count}/{len(combined_payload)} eventów do Kafki.")
        else:
            save_log("producer_buses", "warning", "Pobrano pustą odpowiedź z GPS lub Departures API.")
    except RetryError:
        msg = "Nie udało się pobrać danych po 5 próbach."
        save_log("producer_buses", "error", msg)

# --- Główna część ---
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(job, 'interval', minutes=1)
    scheduler.start()

    print("✅ Buses Producer działa...")
    save_log("producer_buses", "info", "Producer uruchomiony.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        producer.close()
        print("⛔ Buses Producer zatrzymany.")
        save_log("producer_buses", "info", "Producer zatrzymany.")
    except Exception as e:
        save_log("producer_buses", "error", f"Błąd krytyczny: {str(e)}")
    finally:
        scheduler.shutdown()
        producer.close()