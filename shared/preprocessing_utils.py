from shared.db_conn import SessionLocal
from shared.db_utils import get_closest_environment, save_log
import statistics
import datetime

# Funkcja przerabiajaca dane buses w celu ich wypłaszczenia pod Machine Learning
def refactor_buses_data(bus_data):
    refactored = {k: v for k, v in bus_data.items() if k != "estimated_stops"}
    estimated_stops = bus_data.get("estimated_stops", [])
    
    if not estimated_stops:
        refactored.update({
            "stops_count": 0,
            "avg_delay": 0.0,
            "max_delay": 0,
            "min_delay": 0,
            "delay_variance": 0.0,
            "delay_std_dev": 0.0,
            "delay_range": 0,
            "on_time_stops": 0,
            "early_stops": 0,
            "late_stops": 0,
            "delay_trend": "stable",
            "consistency_score": 1.0,
            "punctuality_ratio": 0.0,
            "avg_positive_delay": 0.0,
            "avg_negative_delay": 0.0
        })
        save_log("subscriber_buses","warning", "Brak danych o przystankach.")
        return refactored
    
    delays = [stop["delay"] for stop in estimated_stops if stop.get("delay") is not None]
    
    if not delays:
        refactored.update({
            "stops_count": len(estimated_stops),
            "avg_delay": 0.0,
            "max_delay": 0,
            "min_delay": 0,
            "delay_variance": 0.0,
            "delay_std_dev": 0.0,
            "delay_range": 0,
            "on_time_stops": 0,
            "early_stops": 0,
            "late_stops": 0,
            "delay_trend": "stable",
            "consistency_score": 1.0,
            "punctuality_ratio": 0.0,
            "avg_positive_delay": 0.0,
            "avg_negative_delay": 0.0
        })
        save_log("subscriber_buses","warning", "Brak prawidłowych wartości opóźnień.")
        return refactored

    # Podstawowe statystyki
    avg_delay = statistics.mean(delays)
    max_delay = max(delays)
    min_delay = min(delays)
    delay_variance = statistics.variance(delays) if len(delays) > 1 else 0.0
    delay_std_dev = statistics.stdev(delays) if len(delays) > 1 else 0.0
    delay_range = max_delay - min_delay
    
    # Kategoryzacja opóźnień
    on_time_stops = len([d for d in delays if d == 0])
    early_stops = len([d for d in delays if d < 0])
    late_stops = len([d for d in delays if d > 0])

    # Trendy
    delay_trend = calculate_delay_trend(delays)
    consistency_score = round(1 / (1 + delay_variance), 3) if delay_variance > 0 else 1.0
    punctuality_ratio = round(on_time_stops / len(delays), 3)
    
    # Średnie dla dodatnich i ujemnych opóźnień
    positive_delays = [d for d in delays if d > 0]
    negative_delays = [d for d in delays if d < 0]
    
    avg_positive_delay = round(statistics.mean(positive_delays), 2) if positive_delays else 0.0
    avg_negative_delay = round(statistics.mean(negative_delays), 2) if negative_delays else 0.0
    
    # Dodaj wszystkie statystyki do refactored data
    refactored.update({
        "stops_count": len(estimated_stops),
        "avg_delay": round(avg_delay, 2),
        "max_delay": max_delay,
        "min_delay": min_delay,
        "delay_variance": round(delay_variance, 2),
        "delay_std_dev": round(delay_std_dev, 2),
        "delay_range": delay_range,
        "on_time_stops": on_time_stops,
        "early_stops": early_stops,
        "late_stops": late_stops,
        "delay_trend": delay_trend,
        "consistency_score": consistency_score,
        "punctuality_ratio": punctuality_ratio,
        "avg_positive_delay": avg_positive_delay,
        "avg_negative_delay": avg_negative_delay
    })
    save_log("subscriber_buses","info", "Obliczono dane z przystanków.")
    return refactored

def calculate_delay_trend(delays):
    if len(delays) < 2:
        return "stable"
    
    # Policz ile razy opóźnienie rośnie vs maleje
    increases = 0
    decreases = 0
    
    for i in range(1, len(delays)):
        if delays[i] > delays[i-1]:
            increases += 1
        elif delays[i] < delays[i-1]:
            decreases += 1
    
    # Określ trend na podstawie przewagi
    if increases > decreases * 1.5:
        return "increasing"
    elif decreases > increases * 1.5:
        return "decreasing"
    else:
        return "stable"

# Funkcja wzbogacająca dane bikes/buses o dane środowiskowe (environment)
def enrich_data_with_environment(who: str, data: dict) -> dict:
    db = SessionLocal()
    try:
        target_ts = datetime.datetime.utcfromtimestamp(data["timestamp"])
        env = get_closest_environment(db, target_ts)

        environment_fields = {
            "temperature": None,
            "feelslike": None,
            "humidity": None,
            "wind_kph": None,
            "precip_mm": None,
            "cloud": None,
            "visibility_km": None,
            "uv_index": None,
            "is_day": None,
            "condition": None,
            "pm2_5": None,
            "pm10": None,
            "co": None,
            "no2": None,
            "o3": None,
            "so2": None
        }

        if env:
            environment_fields.update({
                "temperature": float(env.temperature) if env.temperature is not None else None,
                "feelslike": float(env.feelslike) if env.feelslike is not None else None,
                "humidity": env.humidity,
                "wind_kph": float(env.wind_kph) if env.wind_kph is not None else None,
                "precip_mm": float(env.precip_mm) if env.precip_mm is not None else None,
                "cloud": env.cloud,
                "visibility_km": float(env.visibility_km) if env.visibility_km is not None else None,
                "uv_index": float(env.uv_index) if env.uv_index is not None else None,
                "is_day": env.is_day,
                "condition": env.condition,
                "pm2_5": float(env.pm2_5) if env.pm2_5 is not None else None,
                "pm10": float(env.pm10) if env.pm10 is not None else None,
                "co": float(env.co) if env.co is not None else None,
                "no2": float(env.no2) if env.no2 is not None else None,
                "o3": float(env.o3) if env.o3 is not None else None,
                "so2": float(env.so2) if env.so2 is not None else None
            })

            save_log(who, "info", "Udało się wzbogadzić dane o metadane środowiskowe.")
        else:
            save_log(who, "warning", "Brak danych środowiskowych.")
        enriched_data = {**data, **environment_fields}
        return enriched_data

    except Exception as e:
        save_log(who, "error", f"Błąd podczas wzbogacania danych: {str(e)}")
        return data

    finally:
        db.close()

# Funkcja zamieniające nazwy zmiennych słownika na bardziej znajome dla LLM oraz oczyszczające NULLe dla ML
RENAME_MAP = {
    "pm2_5": "fine_particles_pm2_5",
    "pm10": "coarse_particles_pm10",
    "co": "carbon_monoxide_ppb",
    "no2": "nitrogen_dioxide_ppb",
    "o3": "ozone_ppb",
    "so2": "sulfur_dioxide_ppb",
    "bike_available": "manual_bikes_available",
    "ebike_available": "electric_bikes_available",
    "is_day": "daylight",
    "condition": "weather_condition",
    "avg_delay": "average_delay_seconds",
    "max_delay": "maximum_delay_seconds",
    "min_delay": "minimum_delay_seconds",
    "delay_std_dev": "delay_standard_deviation",
    "delay_variance": "delay_variance_value",
    "delay_range": "delay_range_seconds",
    "early_stops": "stops_arrived_early_count",
    "late_stops": "stops_arrived_late_count",
    "on_time_stops": "stops_on_time_count",
    "consistency_score": "delay_consistency_score",
    "punctuality_ratio": "on_time_stop_ratio",
    "avg_positive_delay": "avg_positive_delay_seconds",
    "avg_negative_delay": "avg_negative_delay_seconds",
    "trip_id": "trip_identifier",
    "route_id": "route_identifier",
    "line": "bus_line_number",
    "headsign": "route_destination",
}

def rename_keys(data: dict) -> dict:
    return {
        RENAME_MAP.get(key, key): value
        for key, value in data.items()
    }

def replace_nulls(data: dict) -> dict:
    return {
        key: ("unknown" if value is None else value)
        for key, value in data.items()
    }
