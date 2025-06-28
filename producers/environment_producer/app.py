from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer
import requests
import time
import json

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(60)

# --- Importy połączenia się i funkcji łączących się z PostGreSQL ---
from shared.db_utils import save_log

# --- Ustawienia podstawowe ---
API_KEY = "215f6a06095e40d6b94164258252406"
CITY = "Gdańsk"

KAFKA_BROKER = "kafka-broker-1:9092"
KAFKA_TOPIC = "environment"

# --- Ustawienie Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# --- Funkcje Pomocnicze ---
def get_timestamp():
    return int(datetime.now(timezone.utc).timestamp())

def is_day_to_string(is_day_value):
    return "yes" if is_day_value == 1 else "no"

# --- Funkcja pobierająca dane z WeatherAPI ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=60, min=60, max=300))
def fetch_weatherapi_data(city: str) -> dict:
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        current = data["current"]

        return {
            "type": "environment",
            "timestamp": get_timestamp(),
            "location": {
                "city": city,
                "lat": data["location"]["lat"],
                "lon": data["location"]["lon"]
            },
            "weather": {
                "temperature": current.get("temp_c"),
                "feelslike": current.get("feelslike_c"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "precip_mm": current.get("precip_mm"),
                "cloud": current.get("cloud"),
                "visibility_km": current.get("vis_km"),
                "uv_index": current.get("uv"),
                "is_day": is_day_to_string(current.get("is_day")),
                "condition": current.get("condition", {}).get("text")
            },
            "air_quality": {
                "pm2_5": current.get("air_quality", {}).get("pm2_5"),
                "pm10": current.get("air_quality", {}).get("pm10"),
                "co": current.get("air_quality", {}).get("co"),
                "no2": current.get("air_quality", {}).get("no2"),
                "o3": current.get("air_quality", {}).get("o3"),
                "so2": current.get("air_quality", {}).get("so2")
            }
        }
    else:
        raise Exception(f"API returned status code {response.status_code}")

# --- Funkcja wysyłająca dane do Kafki ---
def send_to_kafka(payload: dict):
    try:
        producer.send(KAFKA_TOPIC, payload)
        producer.flush()
    except Exception as e:
        msg = (f"Błąd wysyłania do Kafki: {str(e)}")
        save_log("producer_environment", "error", msg)

# --- Funkcja wykonująca funkcję pobierania danych z Weather API oraz następnie wysyłająca dane do Kafki ---
def job():
    try:
        payload = fetch_weatherapi_data(CITY)
        if payload:
            send_to_kafka(payload)
            save_log("producer_environment", "info", "Wysłano dane środowiskowe do Kafki.")
        else:
            save_log("producer_environment", "warning", "Pobrano pustą odpowiedź z API.")
    except RetryError:
        msg = "Nie udało się pobrać danych po 5 próbach."
        save_log("producer_environment", "error", msg)

# --- Główna część ---
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(job, 'interval', minutes=2)
    scheduler.start()

    print("✅ Enviroment Producer działa...")
    save_log("producer_environment", "info", "Producer uruchomiony.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        producer.close()
        print("⛔ Enviroment Producer zatrzymany.")
        save_log("producer_environment", "info", "Producer zatrzymany.")
    except Exception as e:
        save_log("producer_environment", "error", f"Błąd krytyczny: {str(e)}")
    finally:
        scheduler.shutdown()
        producer.close()