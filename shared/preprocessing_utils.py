from shared.classification.classification import IS_LATE_MAPPING, DELAY_CATEGORY_MAPPING, BIKE_BINARY_MAPPING, BIKE_MULTICLASS_MAPPING
from shared.db_utils import get_closest_environment, save_log
from shared.db_conn import SessionLocal
from typing import Dict, Any, List
from datetime import datetime, date
import statistics
import json

# +--------------------------------------------------+
# |         PRZYGOTOWANIE DANYCH AUTOBUSOWYCH        |
# |           Funkcja główna i pomocnicza            |
# +--------------------------------------------------+

# Funkcja konwertująca czas timestamp
def format_timestamp_for_display(timestamp_unix: Any) -> str:
    if timestamp_unix is None:
        return "Brak czasu"
    try:
        # Sprawdzamy, czy timestamp jest w milisekundach (jest znacznie większy)
        if timestamp_unix > 10**10: 
            return datetime.fromtimestamp(timestamp_unix / 1000).strftime('%Y-%m-%d %H:%M')
        else: # Zakładamy, że jest w sekundach
            return datetime.fromtimestamp(timestamp_unix).strftime('%Y-%m-%d %H:%M')
    except (TypeError, ValueError):
        return "Nieznany czas"

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

# +--------------------------------------------+
# |         WZBOGACENIE DANYCH POGODĄ          |
# |  Funkcje wzbogacające i oczyszcające dane  |
# +--------------------------------------------+

# Funkcja wzbogacająca dane bikes/buses o dane środowiskowe (environment)
def enrich_data_with_environment(who: str, data: dict) -> dict:
    db = SessionLocal()
    try:
        target_ts = datetime.utcfromtimestamp(data["timestamp"])
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

# Funkcja zamieniające nazwy zmiennych słownika na bardziej znajome dla LLM oraz oczyszczające NULLE dla ML
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

STRING_COLUMNS_TO_HANDLE_AS_UNKNOWN: List[str] = [
    "vehicle_id",
    "bus_line_number",
    "route_destination",
    "station_id",
    "name"
]

NUMERIC_COLUMNS_TO_HANDLE_AS_ZERO: List[str] = [
    "lat",
    "lon",
    "speed",
    "direction",
]

def rename_keys(data: dict) -> dict:
    return {
        RENAME_MAP.get(key, key): value
        for key, value in data.items()
    }

def replace_nulls(data: dict) -> dict:
    processed_data = {}
    for key, value in data.items():
        if value is None:
            if key in STRING_COLUMNS_TO_HANDLE_AS_UNKNOWN:
                processed_data[key] = "unknown"
            elif key in NUMERIC_COLUMNS_TO_HANDLE_AS_ZERO:
                processed_data[key] = 0.0
            else:
                processed_data[key] = None 
        else:
            processed_data[key] = value
            
    return processed_data

# +-------------------------------------+
# |          TWORZENIE ZDANIA           |
# |  Funkcje tworzące zdania z danych   |
# +-------------------------------------+

# Funkcja tworząca zdanie z danych rowerowych:
def create_bike_summary_sentence(data: Dict) -> str:
    summary_parts = []

    # Informacje podstawowe o stacji
    station_name = data.get('name', 'nieznana stacja')
    station_id = data.get('station_id', 'nieznany ID')
    capacity = data.get('capacity')
    
    if station_name and station_id:
        if capacity is not None:
            summary_parts.append(f"Stacja rowerowa '{station_name}' (ID: {station_id}) o pojemności {capacity} miejsc.")
        else:
            summary_parts.append(f"Stacja rowerowa '{station_name}' (ID: {station_id}).")
    elif station_id: # Jeśli nie ma nazwy, ale jest ID
        if capacity is not None:
            summary_parts.append(f"Stacja rowerowa o ID {station_id} o pojemności {capacity} miejsc.")
        else:
            summary_parts.append(f"Stacja rowerowa o ID {station_id}.")
    else:
        summary_parts.append("Nieznana stacja rowerowa.")

    # Dostępność rowerów/doków
    bikes_avail = data.get('bikes_available')
    docks_avail = data.get('docks_available')
    manual_bikes = data.get('manual_bikes_available')
    electric_bikes = data.get('electric_bikes_available')

    if bikes_avail is not None and docks_avail is not None:
        availability_info = f"Aktualnie dostępnych jest {bikes_avail} rowerów (w tym {manual_bikes or 0} manualnych i {electric_bikes or 0} elektrycznych) oraz {docks_avail} wolnych doków."
        summary_parts.append(availability_info)
    elif bikes_avail is not None:
        summary_parts.append(f"Dostępnych rowerów: {bikes_avail}.")
    elif docks_avail is not None:
        summary_parts.append(f"Wolnych doków: {docks_avail}.")

    # Predykcja klastra
    cluster_id = data.get('cluster_id')
    if cluster_id is not None:
        summary_parts.append(f"Stacja została przypisana do klastra {cluster_id}.")

    # Predykcja binarna (teraz z etykietą tekstową)
    bike_binary_label = data.get('bike_binary_label')
    if bike_binary_label:
        summary_parts.append(f"Przewidywany status stacji: {bike_binary_label}.")

    # Predykcja wieloklasowa (teraz z etykietą tekstową)
    bike_multiclass_label = data.get('bike_multiclass_label')
    if bike_multiclass_label:
        summary_parts.append(f"Kategoria dostępności rowerów: {bike_multiclass_label}.")

    # Predykcja regresji (przewidywana liczba rowerów)
    bike_regression_pred = data.get('bike_regression_prediction')
    if bike_regression_pred is not None:
        # Obcięcie do 0, jeśli jest ujemne
        clipped_prediction = max(0, bike_regression_pred)
        summary_parts.append(f"Przewidywana liczba dostępnych rowerów w przyszłości: {clipped_prediction:.0f}.")
        if bike_regression_pred < 0:
            summary_parts.append("(Prognoza była ujemna i została obcięta do zera).")

    # Warunki pogodowe
    weather_cond = data.get('weather_condition')
    temp = data.get('temperature')
    wind_kph = data.get('wind_kph')
    precip_mm = data.get('precip_mm')

    weather_info_parts = []
    if weather_cond:
        weather_info_parts.append(f"warunki pogodowe: {weather_cond}")
    if temp is not None:
        weather_info_parts.append(f"temperatura: {temp:.1f}°C")
    if wind_kph is not None:
        weather_info_parts.append(f"wiatr: {wind_kph:.1f} km/h")
    if precip_mm is not None and precip_mm > 0:
        weather_info_parts.append(f"opady: {precip_mm:.1f} mm")
    
    if weather_info_parts:
        summary_parts.append(f"Aktualne {', '.join(weather_info_parts)}.")

    # Jakość powietrza (opcjonalnie, jeśli chcesz dodać do zdania)
    pm2_5 = data.get('fine_particles_pm2_5')
    pm10 = data.get('coarse_particles_pm10')
    if pm2_5 is not None and pm10 is not None:
        summary_parts.append(f"Jakość powietrza: PM2.5 na poziomie {pm2_5:.1f} µg/m³, PM10 na poziomie {pm10:.1f} µg/m³.")
    save_log("subscriber_bikes", "info", "Stworzono zdanie pod embedding.")

    return " ".join(summary_parts).strip()

# Funkcja tworząca zdanie z danych autobusowych:
def create_bus_summary_sentence(data: Dict) -> str:
    summary_parts = []

    # Informacje podstawowe
    bus_line = data.get('bus_line_number', 'nieznana linia')
    route_dest = data.get('route_destination', 'nieznany cel')
    trip_id = data.get('trip_identifier')
    
    if trip_id is not None:
        summary_parts.append(f"Przejazd autobusu linii {bus_line} (ID podróży: {trip_id}) w kierunku {route_dest}.")
    else:
        summary_parts.append(f"Autobus linii {bus_line} w kierunku {route_dest}.")

    # Predykcja klastra
    cluster_id = data.get('cluster_id')
    if cluster_id is not None:
        summary_parts.append(f"Został przypisany do klastra {cluster_id}.")

    # Predykcja binarna
    is_late_label = data.get('is_late_label')
    if is_late_label:
        summary_parts.append(f"Przewidywany status: {is_late_label}.")

    # Predykcja wieloklasowa
    delay_category_label = data.get('delay_category_label')
    if delay_category_label:
        summary_parts.append(f"Kategoria opóźnienia: {delay_category_label}.")

    # Predykcja regresji
    avg_delay_pred = data.get('average_delay_seconds_prediction')
    if avg_delay_pred is not None:
        summary_parts.append(f"Przewidywane opóźnienie: {avg_delay_pred:.0f} sekund.")
        # Dodatkowa informacja, jeśli predykcja była ujemna i została obcięta
        if data.get('average_delay_seconds_prediction_success') and avg_delay_pred == 0 and data.get('average_delay_seconds_prediction_original', 1) < 0:
            summary_parts.append("(Prognoza była ujemna i została obcięta do zera).")

    # Warunki pogodowe
    weather_cond = data.get('weather_condition')
    temp = data.get('temperature')
    wind_kph = data.get('wind_kph')
    precip_mm = data.get('precip_mm')

    weather_info_parts = []
    if weather_cond:
        weather_info_parts.append(f"warunki pogodowe: {weather_cond}")
    if temp is not None:
        weather_info_parts.append(f"temperatura: {temp:.1f}°C")
    if wind_kph is not None:
        weather_info_parts.append(f"wiatr: {wind_kph:.1f} km/h")
    if precip_mm is not None and precip_mm > 0:
        weather_info_parts.append(f"opady: {precip_mm:.1f} mm")
    
    if weather_info_parts:
        summary_parts.append(f"Aktualne {', '.join(weather_info_parts)}.")

    # Jakość powietrza (opcjonalnie)
    pm2_5 = data.get('fine_particles_pm2_5')
    pm10 = data.get('coarse_particles_pm10')
    if pm2_5 is not None and pm10 is not None:
        summary_parts.append(f"Jakość powietrza: PM2.5 na poziomie {pm2_5:.1f} µg/m³, PM10 na poziomie {pm10:.1f} µg/m³.")
    save_log("subscriber_buses", "info", "Stworzono zdanie pod embedding.")
    
    return " ".join(summary_parts).strip()

# +--------------------------------------------------+
# |          PRZYGOTOWANIE DANYCH POD SQL            |
# |        Funkcje tworzące zdania z danych          |
# +--------------------------------------------------+

# Przygotowuje słownik danych do zapisu w bazie SQL, łącząc wszystkie wzbogacone dane z wygenerowanym zdaniem podsumowującym.
def prepare_sql_record_all_fields(enriched_data: Dict, summary_sentence: str) -> Dict:

    sql_record = enriched_data.copy() # Pracujemy na kopii, aby nie modyfikować oryginalnego enriched_data
    sql_record['summary_sentence'] = summary_sentence
    return sql_record

# +--------------------------------------------------+
# |      PRZYGOTOWANIE DANYCH POD VECTOR DB          |
# |        Funkcje tworzące zdania z danych          |
# +--------------------------------------------------+

# Przygotowuje metadane do wysłania do AnythingLLM z enriched_data dla danych rowerpwych
def prepare_vector_db_record_all_bike_fields(enriched_data: Dict[str, Any]) -> Dict[str, Any]:
    timestamp_display = format_timestamp_for_display(enriched_data.get('timestamp'))

    metadata = {
        "title": f"Dane dla stacji: {enriched_data.get('name', 'Brak nazwy')} (ID: {enriched_data.get('station_id', 'Brak ID')}) - {timestamp_display}",
        "station_id": enriched_data.get("station_id"),
        "station_name": enriched_data.get("name"),
        "timestamp": timestamp_display,
        "bikes_available": enriched_data.get("bikes_available"),
        "capacity": enriched_data.get("capacity"),
        "docks_available": enriched_data.get("docks_available"),
        "manual_bikes_available": enriched_data.get("manual_bikes_available"),
        "electric_bikes_available": enriched_data.get("electric_bikes_available"),
        "temperature": enriched_data.get("temperature"),
        "feelslike": enriched_data.get("feelslike"),
        "humidity": enriched_data.get("humidity"),
        "wind_kph": enriched_data.get("wind_kph"),
        "precip_mm": enriched_data.get("precip_mm"),
        "cloud": enriched_data.get("cloud"),
        "visibility_km": enriched_data.get("visibility_km"),
        "uv_index": enriched_data.get("uv_index"),
        "daylight": enriched_data.get("daylight"),
        "weather_condition": enriched_data.get("weather_condition"),
        "fine_particles_pm2_5": enriched_data.get("fine_particles_pm2_5"),
        "coarse_particles_pm10": enriched_data.get("coarse_particles_pm10"),
        "carbon_monoxide_ppb": enriched_data.get("carbon_monoxide_ppb"),
        "nitrogen_dioxide_ppb": enriched_data.get("nitrogen_dioxide_ppb"),
        "ozone_ppb": enriched_data.get("ozone_ppb"),
        "sulfur_dioxide_ppb": enriched_data.get("sulfur_dioxide_ppb"),
        "cluster_id": enriched_data.get("cluster_id"),
        "cluster_prediction_success": enriched_data.get("cluster_prediction_success"),
        "bike_binary_prediction": enriched_data.get("bike_binary_prediction"),
        "bike_binary_probabilities": enriched_data.get("bike_binary_probabilities"),
        "bike_binary_prediction_success": enriched_data.get("bike_binary_prediction_success"),
        "bike_binary_label": enriched_data.get("bike_binary_label"),
        "bike_multiclass_prediction": enriched_data.get("bike_multiclass_prediction"),
        "bike_multiclass_probabilities": enriched_data.get("bike_multiclass_probabilities"),
        "bike_multiclass_prediction_success": enriched_data.get("bike_multiclass_prediction_success"),
        "bike_multiclass_label": enriched_data.get("bike_multiclass_label"),
        "bike_regression_prediction": enriched_data.get("bike_regression_prediction"),
        "bike_regression_prediction_original": enriched_data.get("bike_regression_prediction_original"),
        "bike_regression_prediction_success": enriched_data.get("bike_regression_prediction_success"),
    }
    
    # Konwertuj wartości, które mogą być obiektami na stringi, aby uniknąć błędów serializacji JSON
    for key, value in metadata.items():
        if isinstance(value, (datetime, date)):
            metadata[key] = value.isoformat()
        elif isinstance(value, (dict, list)):
             metadata[key] = json.dumps(value)
        elif value is None:
            metadata[key] = "N/A"

    return metadata

# Przygotowuje metadane do wysłania do AnythingLLM z enriched_data dla danych autobusowych.
def prepare_vector_db_record_all_bus_fields(enriched_data: Dict[str, Any]) -> Dict[str, Any]:
    timestamp_display = format_timestamp_for_display(enriched_data.get('timestamp'))

    metadata = {
        "title": f"Autobus Linii {enriched_data.get('bus_line_number', 'Brak Linii')} ({enriched_data.get('vehicle_id', 'Brak ID')}) - Cel: {enriched_data.get('route_destination', 'Brak Celu')} - {timestamp_display}",
        "vehicle_id": enriched_data.get("vehicle_id"),
        "bus_line_number": enriched_data.get("bus_line_number"),
        "timestamp": timestamp_display,
        "route_destination": enriched_data.get("route_destination"),
        "lat": enriched_data.get("lat"),
        "lon": enriched_data.get("lon"),
        "speed": enriched_data.get("speed"),
        "direction": enriched_data.get("direction"),
        "trip_identifier": enriched_data.get("trip_identifier"),
        "route_identifier": enriched_data.get("route_identifier"),
        "stops_count": enriched_data.get("stops_count"),
        "average_delay_seconds": enriched_data.get("average_delay_seconds"),
        "maximum_delay_seconds": enriched_data.get("maximum_delay_seconds"),
        "minimum_delay_seconds": enriched_data.get("minimum_delay_seconds"),
        "delay_variance_value": enriched_data.get("delay_variance_value"),
        "delay_standard_deviation": enriched_data.get("delay_standard_deviation"),
        "delay_range_seconds": enriched_data.get("delay_range_seconds"),
        "stops_on_time_count": enriched_data.get("stops_on_time_count"),
        "stops_arrived_early_count": enriched_data.get("stops_arrived_early_count"),
        "stops_arrived_late_count": enriched_data.get("stops_arrived_late_count"),
        "delay_trend": enriched_data.get("delay_trend"),
        "delay_consistency_score": enriched_data.get("delay_consistency_score"),
        "on_time_stop_ratio": enriched_data.get("on_time_stop_ratio"),
        "avg_positive_delay_seconds": enriched_data.get("avg_positive_delay_seconds"),
        "avg_negative_delay_seconds": enriched_data.get("avg_negative_delay_seconds"),
        "temperature": enriched_data.get("temperature"),
        "feelslike": enriched_data.get("feelslike"),
        "humidity": enriched_data.get("humidity"),
        "wind_kph": enriched_data.get("wind_kph"),
        "precip_mm": enriched_data.get("precip_mm"),
        "cloud": enriched_data.get("cloud"),
        "visibility_km": enriched_data.get("visibility_km"),
        "uv_index": enriched_data.get("uv_index"),
        "daylight": enriched_data.get("daylight"),
        "weather_condition": enriched_data.get("weather_condition"),
        "fine_particles_pm2_5": enriched_data.get("fine_particles_pm2_5"),
        "coarse_particles_pm10": enriched_data.get("coarse_particles_pm10"),
        "carbon_monoxide_ppb": enriched_data.get("carbon_monoxide_ppb"),
        "nitrogen_dioxide_ppb": enriched_data.get("nitrogen_dioxide_ppb"),
        "ozone_ppb": enriched_data.get("ozone_ppb"),
        "sulfur_dioxide_ppb": enriched_data.get("sulfur_dioxide_ppb"),
        "cluster_id": enriched_data.get("cluster_id"),
        "cluster_prediction_success": enriched_data.get("cluster_prediction_success"),
        "is_late_prediction": enriched_data.get("is_late_prediction"),
        "is_late_probabilities": enriched_data.get("is_late_probabilities"),
        "is_late_prediction_success": enriched_data.get("is_late_prediction_success"),
        "is_late_label": enriched_data.get("is_late_label"),
        "delay_category_prediction": enriched_data.get("delay_category_prediction"),
        "delay_category_probabilities": enriched_data.get("delay_category_probabilities"),
        "delay_category_prediction_success": enriched_data.get("delay_category_prediction_success"),
        "delay_category_label": enriched_data.get("delay_category_label"),
        "average_delay_seconds_prediction": enriched_data.get("average_delay_seconds_prediction"),
        "average_delay_seconds_prediction_original": enriched_data.get("average_delay_seconds_prediction_original"),
        "average_delay_seconds_prediction_success": enriched_data.get("average_delay_seconds_prediction_success"),
    }
    
    # Konwertuj wartości, które mogą być obiektami na stringi, aby uniknąć błędów serializacji JSON
    for key, value in metadata.items():
        if isinstance(value, (datetime, date)):
            metadata[key] = value.isoformat()
        elif isinstance(value, (dict, list)):
            metadata[key] = json.dumps(value) 
        elif value is None:
            metadata[key] = "N/A"

    return metadata
