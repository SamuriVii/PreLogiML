from shared.db_dto import BikesData, BusesData, CEST
from datetime import datetime, timedelta, timezone
from shared.db_conn import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd

# +-------------------------------------+
# |  FUNKCJE WSPOMAGAJĄCE GROUND TRUTH  |
# |      Generowanie pytań dla LLM      |
# +-------------------------------------+

# Ground Truth dla pytania o klasteryzację stacji rowerowej, sukces predykcji i opady.
def get_gt_bike_station_cluster_status(session: Session, station_name: str, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)
    
    data = session.query(
        BikesData.cluster_id,
        BikesData.cluster_prediction_success,
        BikesData.precip_mm
    ).filter(
        BikesData.name == station_name,
        BikesData.timestamp >= start_time
    ).order_by(BikesData.timestamp.desc()).first()

    if data:
        precip_status = "tak" if data.precip_mm and data.precip_mm > 0 else "nie"
        return (f"Dla stacji rowerowej '{station_name}': Id klastra to {data.cluster_id}. "
                f"Predykcja klastra była {'udana' if data.cluster_prediction_success else 'nieudana'}. "
                f"W tym czasie występowały silne opady deszczu: {precip_status}.")
    return f"Brak danych dla stacji rowerowej '{station_name}' w ciągu ostatnich {time_interval_hours} godzin."

# Ground Truth dla pytania o autobusy w klastrze i dominującą kategorię opóźnienia.
def get_gt_bus_cluster_delay_category(session: Session, cluster_id: int, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    results = session.query(
        BusesData.vehicle_id,
        BusesData.delay_category_label
    ).filter(
        BusesData.cluster_id == cluster_id,
        BusesData.timestamp >= start_time
    ).all()

    if not results:
        return f"Brak danych dla klastra o id '{cluster_id}' w ciągu ostatnich {time_interval_hours} godzin."

    vehicle_ids = set()
    delay_labels = {}
    for r in results:
        vehicle_ids.add(r.vehicle_id)
        if r.delay_category_label:
            delay_labels[r.delay_category_label] = delay_labels.get(r.delay_category_label, 0) + 1
    
    dominant_label = max(delay_labels, key=delay_labels.get) if delay_labels else "brak danych"
    
    return (f"W ciągu ostatnich {time_interval_hours} godzin, {len(vehicle_ids)} unikalnych autobusów należało do klastra o id '{cluster_id}'. "
            f"Dominująca kategoria opóźnienia dla tych autobusów to: '{dominant_label}'.")

# Ground Truth dla pytania o średnią liczbę rowerów elektrycznych dla udanych predykcji binarnych.
def get_gt_avg_electric_bikes_success_binary(session: Session, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    avg_bikes = session.query(
        func.avg(BikesData.electric_bikes_available)
    ).filter(
        BikesData.bike_binary_prediction_success == True,
        BikesData.timestamp >= start_time
    ).scalar()

    if avg_bikes is not None:
        return (f"Średnia liczba dostępnych rowerów elektrycznych dla wszystkich pomyślnych predykcji binarnych "
                f"w ciągu ostatnich {time_interval_hours} godzin wynosi: {avg_bikes:.2f}.")
    return f"Brak danych o pomyślnych predykcjach binarnych rowerów w ciągu ostatnich {time_interval_hours} godzin."

# Ground Truth dla pytania o najczęściej występującą etykietę opóźnienia dla pomyślnych predykcji + on_time_stop_ratio.
def get_gt_most_common_delay_label_success(session: Session, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    # Znajdź najczęściej występującą etykietę
    most_common_label_query = session.query(
        BusesData.delay_category_label,
        func.count(BusesData.delay_category_label)
    ).filter(
        BusesData.delay_category_prediction_success == True,
        BusesData.delay_category_label.isnot(None),
        BusesData.timestamp >= start_time
    ).group_by(BusesData.delay_category_label).order_by(func.count(BusesData.delay_category_label).desc()).first()

    if not most_common_label_query:
        return f"Brak danych o pomyślnych predykcjach opóźnień autobusów w ciągu ostatnich {time_interval_hours} godzin."

    most_common_label = most_common_label_query[0]

    # Oblicz średnie on_time_stop_ratio dla tej etykiety
    avg_ratio = session.query(
        func.avg(BusesData.on_time_stop_ratio)
    ).filter(
        BusesData.delay_category_prediction_success == True,
        BusesData.delay_category_label == most_common_label,
        BusesData.timestamp >= start_time
    ).scalar()

    return (f"Najczęściej występująca etykieta opóźnienia dla pomyślnych predykcji w ciągu ostatnich {time_interval_hours} godzin to: '{most_common_label}'. "
            f"Średnia wartość 'on_time_stop_ratio' dla tej etykiety wynosi: {avg_ratio:.2f}.")

# Ground Truth dla pytania o warunki pogodowe dla niskiej dostępności rowerów w klastrze.
def get_gt_weather_low_bike_availability_cluster(session: Session, cluster_id: int, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    weather_conditions_query = session.query(
        BikesData.weather_condition,
        func.count(BikesData.weather_condition)
    ).filter(
        BikesData.cluster_id == cluster_id,
        BikesData.bikes_available < 5,
        BikesData.weather_condition.isnot(None),
        BikesData.timestamp >= start_time
    ).group_by(BikesData.weather_condition).order_by(func.count(BikesData.weather_condition).desc()).first()

    if weather_conditions_query:
        return (f"Dla stacji rowerowych w klastrze o id '{cluster_id}', które miały niską dostępność rowerów "
                f"w ciągu ostatnich {time_interval_hours} godzin, najczęściej występującym warunkiem pogodowym było: '{weather_conditions_query[0]}'.")
    return f"Brak danych dla klastra o id '{cluster_id}' z niską dostępnością rowerów w ciągu ostatnich {time_interval_hours} godzin."

# Ground Truth dla pytania o korelację PM2.5 a średnie opóźnienie autobusów dla linii.
def get_gt_bus_pm25_delay_correlation(session: Session, bus_line_number: str, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    data = session.query(
        BusesData.fine_particles_pm2_5,
        BusesData.delay_category_prediction_success
    ).filter(
        BusesData.bus_line_number == bus_line_number,
        BusesData.timestamp >= start_time
    ).all()

    if not data:
        return f"Brak danych dla linii autobusowej '{bus_line_number}' w ciągu ostatnich {time_interval_hours} godzin."

    successful_pm25 = [d.fine_particles_pm2_5 for d in data if d.delay_category_prediction_success and d.fine_particles_pm2_5 is not None]
    unsuccessful_pm25 = [d.fine_particles_pm2_5 for d in data if not d.delay_category_prediction_success and d.fine_particles_pm2_5 is not None]

    avg_successful = sum(successful_pm25) / len(successful_pm25) if successful_pm25 else 0
    avg_unsuccessful = sum(unsuccessful_pm25) / len(unsuccessful_pm25) if unsuccessful_pm25 else 0

    correlation_desc = ""
    if avg_successful > avg_unsuccessful and avg_unsuccessful != 0:
        correlation_desc = "Tak, średnie stężenie PM2.5 dla pomyślnych predykcji było wyższe."
    elif avg_successful < avg_unsuccessful and avg_successful != 0:
        correlation_desc = "Tak, średnie stężenie PM2.5 dla pomyślnych predykcji było niższe."
    else:
        correlation_desc = "Nie zaobserwowano wyraźnej różnicy w średnim stężeniu PM2.5 między pomyślnymi a niepomyślnymi predykcjami."

    return (f"{correlation_desc} "
            f"Średnie stężenie PM2.5 dla pomyślnych predykcji: {avg_successful:.2f}. "
            f"Dla niepomyślnych predykcji: {avg_unsuccessful:.2f}.")

# Ground Truth dla pytania o wpływ wilgotności i temperatury na prognozowaną liczbę rowerów.
def get_gt_humidity_temp_bike_prediction_impact(session: Session, station_name: str, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    data = session.query(
        BikesData.humidity,
        BikesData.temperature,
        BikesData.bike_regression_prediction
    ).filter(
        BikesData.name == station_name,
        BikesData.timestamp >= start_time,
        BikesData.humidity.isnot(None),
        BikesData.temperature.isnot(None),
        BikesData.bike_regression_prediction.isnot(None)
    ).all()

    if not data or len(data) < 2: # Potrzeba co najmniej 2 punktów do analizy trendu
        return f"Brak wystarczających danych dla stacji '{station_name}' w ciągu ostatnich {time_interval_hours} godzin do analizy wpływu wilgotności i temperatury."

    df = pd.DataFrame([(d.humidity, d.temperature, d.bike_regression_prediction) for d in data],
                      columns=['humidity', 'temperature', 'bike_prediction'])

    # Prosta analiza korelacji
    corr_humidity = df['humidity'].corr(df['bike_prediction'])
    corr_temp = df['temperature'].corr(df['bike_prediction'])

    impact_desc = []
    if pd.notna(corr_humidity):
        if corr_humidity > 0.3:
            impact_desc.append(f"Wzrost wilgotności ma tendencję do pozytywnego wpływu (korelacja {corr_humidity:.2f}) na prognozowaną liczbę rowerów.")
        elif corr_humidity < -0.3:
            impact_desc.append(f"Wzrost wilgotności ma tendencję do negatywnego wpływu (korelacja {corr_humidity:.2f}) na prognozowaną liczbę rowerów.")
        else:
            impact_desc.append(f"Wilgotność ma słaby wpływ (korelacja {corr_humidity:.2f}) na prognozowaną liczbę rowerów.")
    
    if pd.notna(corr_temp):
        if corr_temp > 0.3:
            impact_desc.append(f"Wzrost temperatury ma tendencję do pozytywnego wpływu (korelacja {corr_temp:.2f}) na prognozowaną liczbę rowerów.")
        elif corr_temp < -0.3:
            impact_desc.append(f"Wzrost temperatury ma tendencję do negatywnego wpływu (korelacja {corr_temp:.2f}) na prognozowaną liczbę rowerów.")
        else:
            impact_desc.append(f"Temperatura ma słaby wpływ (korelacja {corr_temp:.2f}) na prognozowaną liczbę rowerów.")

    return f"Dla stacji '{station_name}': {' '.join(impact_desc) if impact_desc else 'Brak wyraźnego wpływu wilgotności i temperatury na prognozowaną liczbę rowerów.'}"

# Ground Truth dla pytania o warunki widoczności a opóźnienia autobusów 'Very Late'.
def get_gt_visibility_very_late_buses(session: Session, bus_line_number: str, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)

    visibility_query = session.query(
        BusesData.visibility_km,
        func.count(BusesData.visibility_km)
    ).filter(
        BusesData.bus_line_number == bus_line_number,
        BusesData.is_late_label == 'Very Late',
        BusesData.visibility_km.isnot(None),
        BusesData.timestamp >= start_time
    ).group_by(BusesData.visibility_km).order_by(func.count(BusesData.visibility_km).desc()).first()

    if visibility_query:
        return (f"Autobusy linii '{bus_line_number}' najczęściej doświadczały opóźnień sklasyfikowanych jako 'Very Late' "
                f"przy widoczności wynoszącej około {visibility_query[0]:.1f} km w ciągu ostatnich {time_interval_hours} godzin.")
    return f"Brak danych o opóźnieniach 'Very Late' dla linii '{bus_line_number}' w ciągu ostatnich {time_interval_hours} godzin."

# Ground Truth dla pytania o wskaźnik UV, rowery manualne vs elektryczne w klastrze, wpływ na opóźnienie autobusów.
# Uproszczone "sąsiadujące obszary" do "tego samego dnia i podobnych warunków pogodowych".
def get_gt_uv_bikes_bus_delay_correlation(session: Session, cluster_id: int, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    bike_data = session.query(
        BikesData.timestamp,
        BikesData.uv_index,
        BikesData.manual_bikes_available,
        BikesData.electric_bikes_available,
        BikesData.weather_condition
    ).filter(
        BikesData.cluster_id == cluster_id,
        BikesData.uv_index.isnot(None),
        BikesData.timestamp >= start_time
    ).all()

    bus_data = session.query(
        BusesData.timestamp,
        BusesData.average_delay_seconds,
        BusesData.weather_condition
    ).filter(
        BusesData.timestamp >= start_time,
        BusesData.average_delay_seconds.isnot(None)
    ).all()

    if not bike_data or not bus_data:
        return f"Brak wystarczających danych dla analizy korelacji UV, rowerów i opóźnień autobusów w ciągu ostatnich {time_interval_days} dni."

    bike_df = pd.DataFrame([(d.timestamp.date(), d.uv_index, d.manual_bikes_available, d.electric_bikes_available, d.weather_condition) for d in bike_data],
                           columns=['date', 'uv_index', 'manual_bikes', 'electric_bikes', 'weather_condition'])
    bus_df = pd.DataFrame([(d.timestamp.date(), d.average_delay_seconds, d.weather_condition) for d in bus_data],
                          columns=['date', 'avg_delay', 'weather_condition'])

    high_uv_days_bikes = bike_df[bike_df['uv_index'] > 7]
    
    bike_trend = "Brak wyraźnego trendu."
    if not high_uv_days_bikes.empty:
        avg_manual = high_uv_days_bikes['manual_bikes'].mean()
        avg_electric = high_uv_days_bikes['electric_bikes'].mean()
        if avg_manual > avg_electric:
            bike_trend = f"W dniach o wysokim UV, stacje w klastrze '{cluster_id}' miały średnio więcej rowerów manualnych ({avg_manual:.1f}) niż elektrycznych ({avg_electric:.1f})."
        else:
            bike_trend = f"W dniach o wysokim UV, stacje w klastrze '{cluster_id}' miały średnio więcej rowerów elektrycznych ({avg_electric:.1f}) niż manualnych ({avg_manual:.1f})."

    # Analiza wpływu na autobusy
    merged_df = pd.merge(high_uv_days_bikes, bus_df, on=['date', 'weather_condition'], how='inner')
    bus_impact = "Brak danych o wpływie na opóźnienia autobusów w sąsiadujących obszarach."
    if not merged_df.empty:
        avg_delay_high_uv = merged_df['avg_delay'].mean()
        bus_impact = f"W dniach o wysokim UV i podobnych warunkach pogodowych, średnie opóźnienie autobusów wynosiło {avg_delay_high_uv:.1f} sekund."

    return (f"{bike_trend} "
            f"{bus_impact}")

# Ground Truth dla pytania o optymalne godziny dla dostępności rowerów i opóźnień autobusów.
def get_gt_optimal_hours_bikes_buses(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    bike_data = session.query(
        func.date_trunc('hour', BikesData.timestamp).label('hour_of_day'),
        func.avg(BikesData.bikes_available).label('avg_bikes_available'),
        func.avg(BikesData.cloud).label('avg_cloud_bikes')
    ).filter(
        BikesData.timestamp >= start_time,
        BikesData.cloud.isnot(None)
    ).group_by('hour_of_day').all()

    bus_data = session.query(
        func.date_trunc('hour', BusesData.timestamp).label('hour_of_day'),
        func.avg(BusesData.average_delay_seconds).label('avg_delay_seconds'),
        func.avg(BusesData.cloud).label('avg_cloud_buses')
    ).filter(
        BusesData.timestamp >= start_time,
        BusesData.cloud.isnot(None)
    ).group_by('hour_of_day').all()

    if not bike_data or not bus_data:
        return f"Brak wystarczających danych do analizy optymalnych godzin w ciągu ostatnich {time_interval_days} dni."

    bike_df = pd.DataFrame(bike_data)
    bus_df = pd.DataFrame(bus_data)

    merged_df = pd.merge(bike_df, bus_df, on='hour_of_day', how='inner')
    merged_df['hour'] = merged_df['hour_of_day'].apply(lambda x: x.hour)

    filtered_df = merged_df[(merged_df['avg_cloud_bikes'] < 20) & (merged_df['avg_cloud_buses'] < 20)]

    if filtered_df.empty:
        return f"Brak godzin spełniających kryteria niskiego zachmurzenia w ciągu ostatnich {time_interval_days} dni."

    # --- POPRAWKA TUTAJ: Sprawdzenie, czy zakres nie jest zerowy przed dzieleniem ---
    bikes_range = filtered_df['avg_bikes_available'].max() - filtered_df['avg_bikes_available'].min()
    delay_range = filtered_df['avg_delay_seconds'].max() - filtered_df['avg_delay_seconds'].min()

    filtered_df['norm_bikes'] = 0.0 # Domyślna wartość
    if bikes_range > 0:
        filtered_df['norm_bikes'] = (filtered_df['avg_bikes_available'] - filtered_df['avg_bikes_available'].min()) / bikes_range
    else:
        print("Brak zmienności w avg_bikes_available dla normalizacji. Domyślnie norm_bikes = 0.")

    filtered_df['norm_delay'] = 0.0 # Domyślna wartość
    if delay_range > 0:
        filtered_df['norm_delay'] = (filtered_df['avg_delay_seconds'].max() - filtered_df['avg_delay_seconds']) / delay_range
    else:
        print("Brak zmienności w avg_delay_seconds dla normalizacji. Domyślnie norm_delay = 0.")
    # --- KONIEC POPRAWKI ---

    filtered_df['score'] = filtered_df['norm_bikes'] + filtered_df['norm_delay']

    if filtered_df.empty:
        return f"Brak danych do obliczenia optymalnej godziny po normalizacji w ciągu ostatnich {time_interval_days} dni."

    best_hour_data = filtered_df.sort_values(by='score', ascending=False).iloc[0]

    return (f"W ciągu ostatnich {time_interval_days} dni, przy niskim zachmurzeniu, "
            f"optymalna godzina dobowa to około {best_hour_data['hour']}:00. "
            f"W tym czasie średnia dostępność rowerów to {best_hour_data['avg_bikes_available']:.1f}, "
            f"a średnie opóźnienie autobusów to {best_hour_data['avg_delay_seconds']:.1f} sekund.")

# Ground Truth dla pytania o trzy główne czynniki środowiskowe wpływające na sukces klasyfikacji opóźnień autobusów.
# (Heurystyczne, bazujące na prostej korelacji)
def get_gt_top_environmental_factors_bus_delay_success(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    data = session.query(
        BusesData.delay_category_prediction_success,
        BusesData.temperature,
        BusesData.humidity,
        BusesData.wind_kph,
        BusesData.precip_mm,
        BusesData.cloud,
        BusesData.visibility_km,
        BusesData.uv_index,
        BusesData.fine_particles_pm2_5,
        BusesData.coarse_particles_pm10,
        BusesData.carbon_monoxide_ppb,
        BusesData.nitrogen_dioxide_ppb,
        BusesData.ozone_ppb,
        BusesData.sulfur_dioxide_ppb
    ).filter(
        BusesData.timestamp >= start_time,
        BusesData.delay_category_prediction_success.isnot(None)
    ).all()

    if not data:
        return f"Brak danych do analizy czynników środowiskowych wpływających na sukces predykcji opóźnień autobusów w ciągu ostatnich {time_interval_days} dni."

    df = pd.DataFrame([d._asdict() for d in data])
    df_filtered = df.dropna(subset=['delay_category_prediction_success'])

    if df_filtered.empty:
        return "Brak danych z pomyślnymi/niepomyślnymi predykcjami opóźnień do analizy."

    df_filtered['delay_success_int'] = df_filtered['delay_category_prediction_success'].astype(int)

    environmental_cols = [
        'temperature', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'visibility_km', 'uv_index',
        'fine_particles_pm2_5', 'coarse_particles_pm10', 'carbon_monoxide_ppb',
        'nitrogen_dioxide_ppb', 'ozone_ppb', 'sulfur_dioxide_ppb'
    ]

    correlations = {}
    for col in environmental_cols:
        if col in df_filtered.columns and df_filtered[col].notna().any():
            # --- DODANO SPRAWDZENIE ZMIENNOŚCI DANYCH ---
            if df_filtered[col].nunique() > 1 and df_filtered['delay_success_int'].nunique() > 1:
                correlations[col] = df_filtered['delay_success_int'].corr(df_filtered[col])
            else:
                print(f"Brak zmienności w kolumnie '{col}' lub 'delay_success_int' dla korelacji.")
                pass
            # --- KONIEC DODANEGO SPRAWDZENIA ---

    correlations = {k: v for k, v in correlations.items() if pd.notna(v)}

    if not correlations:
        return "Brak wyraźnych korelacji między czynnikami środowiskowymi a sukcesem predykcji opóźnień autobusów."

    sorted_factors = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    top_3_factors = sorted_factors[:3]

    if not top_3_factors:
        return "Brak wyraźnych czynników środowiskowych wpływających na sukces predykcji opóźnień autobusów."

    result_str = "Trzy główne czynniki środowiskowe, które najbardziej wpływają na prognozowany sukces klasyfikacji opóźnień autobusów to: "
    for i, (factor, corr_val) in enumerate(top_3_factors):
        impact = "pozytywny" if corr_val > 0 else "negatywny"
        result_str += f"{i+1}. {factor} (korelacja: {corr_val:.2f}, wpływ: {impact}). "
    return result_str.strip()

# Ground Truth dla pytania o porównanie sukcesu predykcji klasteryzacji między rowerami i autobusami + średnia temperatura.
def get_gt_cluster_success_comparison_temp(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    # Rowerowe
    bike_success_count = session.query(func.count(BikesData.id)).filter(
        BikesData.cluster_prediction_success == True,
        BikesData.timestamp >= start_time
    ).scalar()
    bike_total_count = session.query(func.count(BikesData.id)).filter(
        BikesData.timestamp >= start_time
    ).scalar()
    bike_avg_temp_success = session.query(func.avg(BikesData.temperature)).filter(
        BikesData.cluster_prediction_success == True,
        BikesData.timestamp >= start_time
    ).scalar()

    bike_success_rate = (bike_success_count / bike_total_count) * 100 if bike_total_count else 0

    # Autobusowe
    bus_success_count = session.query(func.count(BusesData.id)).filter(
        BusesData.cluster_prediction_success == True,
        BusesData.timestamp >= start_time
    ).scalar()
    bus_total_count = session.query(func.count(BusesData.id)).filter(
        BusesData.timestamp >= start_time
    ).scalar()
    bus_avg_temp_success = session.query(func.avg(BusesData.temperature)).filter(
        BusesData.cluster_prediction_success == True,
        BusesData.timestamp >= start_time
    ).scalar()

    bus_success_rate = (bus_success_count / bus_total_count) * 100 if bus_total_count else 0

    result_str = f"W ciągu ostatnich {time_interval_days} dni: "
    result_str += f"Dla stacji rowerowych, wskaźnik sukcesu predykcji klasteryzacji to {bike_success_rate:.2f}%. "
    result_str += f"Średnia temperatura dla pomyślnych predykcji rowerowych: {bike_avg_temp_success:.1f}°C. "
    result_str += f"Dla linii autobusowych, wskaźnik sukcesu predykcji klasteryzacji to {bus_success_rate:.2f}%. "
    result_str += f"Średnia temperatura dla pomyślnych predykcji autobusowych: {bus_avg_temp_success:.1f}°C. "

    if bike_success_rate > bus_success_rate:
        result_str += "Sektor rowerowy wykazał wyższy wskaźnik sukcesu klasteryzacji."
    elif bus_success_rate > bike_success_rate:
        result_str += "Sektor autobusowy wykazał wyższy wskaźnik sukcesu klasteryzacji."
    else:
        result_str += "Wskaźniki sukcesu klasteryzacji w obu sektorach były podobne."
    
    return result_str

# Ground Truth dla pytania o stację rowerową wymagającą uzupełnienia rowerów na podstawie predykcji i pogody.
def get_gt_bike_station_needs_refill(session: Session, time_interval_hours: int = 6) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_hours)
    
    # Definicja "niesprzyjającej pogody"
    unfavorable_weather_conditions = ['Light Rain', 'Moderate Rain', 'Heavy Rain', 'Snow', 'Sleet', 'Freezing Drizzle', 'Blizzard', 'Fog', 'Mist']

    data = session.query(
        BikesData.name,
        BikesData.bikes_available,
        BikesData.bike_regression_prediction,
        BikesData.weather_condition,
        BikesData.timestamp
    ).filter(
        BikesData.bikes_available < 5, # Niska dostępność
        BikesData.weather_condition.in_(unfavorable_weather_conditions), # Niesprzyjająca pogoda
        BikesData.timestamp >= start_time
    ).order_by(BikesData.timestamp.desc()).all()

    if not data:
        return f"Brak stacji wymagających uzupełnienia rowerów w ciągu ostatnich {time_interval_hours} godzin, spełniających kryteria niskiej dostępności i niesprzyjającej pogody."

    # Możemy wybrać stację, która najczęściej spełnia te kryteria lub po prostu pierwszą znalezioną
    station_counts = {}
    for d in data:
        station_counts[d.name] = station_counts.get(d.name, 0) + 1
    
    most_critical_station = max(station_counts, key=station_counts.get)
    
    # Pobierz ostatnie dane dla tej stacji
    last_data_for_station = session.query(BikesData).filter(
        BikesData.name == most_critical_station,
        BikesData.timestamp >= start_time
    ).order_by(BikesData.timestamp.desc()).first()

    if last_data_for_station:
        return (f"Stacja '{last_data_for_station.name}' najprawdopodobniej wymagała uzupełnienia rowerów w ciągu ostatnich {time_interval_hours} godzin. "
                f"Ostatnia zarejestrowana dostępność rowerów: {last_data_for_station.bikes_available}, "
                f"prognozowana dostępność: {last_data_for_station.bike_regression_prediction:.1f}, "
                f"warunki pogodowe: '{last_data_for_station.weather_condition}'.")
    return "Brak danych spełniających kryteria dla rekomendacji uzupełnienia rowerów."

# Ground Truth dla pytania o trasy autobusowe podatne na duże opóźnienia przy dużym deszczu.
def get_gt_bus_routes_high_delay_heavy_rain(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    heavy_rain_conditions = ['Heavy Rain', 'Torrential Rain', 'Moderate Rain', 'Light Rain'] # Rozszerzona definicja "dużego deszczu"

    routes_query = session.query(
        BusesData.bus_line_number,
        func.count(BusesData.bus_line_number)
    ).filter(
        BusesData.average_delay_seconds > 300, # Ponad 5 minut opóźnienia
        BusesData.precip_mm > 5, # Znaczące opady
        BusesData.weather_condition.in_(heavy_rain_conditions),
        BusesData.delay_category_prediction_success == True, # Zgodnie z klasyfikacją ML
        BusesData.timestamp >= start_time
    ).group_by(BusesData.bus_line_number).order_by(func.count(BusesData.bus_line_number).desc()).limit(3).all()

    if routes_query:
        result_str = f"W ciągu ostatnich {time_interval_days} dni, trasy autobusowe najbardziej podatne na duże opóźnienia przy znaczących opadach deszczu (zgodnie z klasyfikacjami ML) to: "
        for i, (line, count) in enumerate(routes_query):
            result_str += f"{line} (liczba zdarzeń: {count}). "
        return result_str.strip()
    return f"Brak danych o trasach autobusowych spełniających kryteria dużych opóźnień przy deszczu w ciągu ostatnich {time_interval_days} dni."

# Ground Truth dla pytania o wzrost zanieczyszczeń a spadek wypożyczeń rowerów (wzrost docks_available).
def get_gt_pollution_bike_rental_correlation(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    data = session.query(
        BikesData.timestamp,
        BikesData.station_id,
        BikesData.fine_particles_pm2_5,
        BikesData.docks_available
    ).filter(
        BikesData.timestamp >= start_time,
        BikesData.fine_particles_pm2_5.isnot(None),
        BikesData.docks_available.isnot(None)
    ).order_by(BikesData.timestamp).all()

    if not data or len(data) < 2:
        return f"Brak wystarczających danych do analizy korelacji zanieczyszczeń i dostępności doków w ciągu ostatnich {time_interval_days} dni."

    df = pd.DataFrame([(d.timestamp, d.station_id, d.fine_particles_pm2_5, d.docks_available) for d in data],
                      columns=['timestamp', 'station_id', 'pm25', 'docks_available'])
    
    # Obliczanie zmiany docks_available
    df['prev_docks'] = df.groupby('station_id')['docks_available'].shift(1)
    df['docks_change'] = df['docks_available'] - df['prev_docks']

    # Definicja "gwałtownego wzrostu docks_available" (np. wzrost o co najmniej 5)
    sudden_increase_threshold = 5 
    sudden_increase_df = df[df['docks_change'] > sudden_increase_threshold]

    if sudden_increase_df.empty:
        return "Brak zdarzeń gwałtownego wzrostu dostępności doków (spadku wypożyczeń) w analizowanym okresie."

    avg_pm25_during_increase = sudden_increase_df['pm25'].mean()
    
    # Prosta ocena korelacji
    overall_corr = df['pm25'].corr(df['docks_available'])

    correlation_desc = ""
    if pd.notna(overall_corr):
        if overall_corr > 0.3:
            correlation_desc = "Tak, istnieje pozytywna korelacja (wzrost zanieczyszczeń koreluje ze wzrostem dostępności doków, czyli spadkiem wypożyczeń)."
        elif overall_corr < -0.3:
            correlation_desc = "Nie, istnieje negatywna korelacja (wzrost zanieczyszczeń koreluje ze spadkiem dostępności doków, czyli wzrostem wypożyczeń)."
        else:
            correlation_desc = "Brak wyraźnej korelacji między wzrostem zanieczyszczeń a dostępnością doków."
    else:
        correlation_desc = "Brak wystarczających danych do obliczenia korelacji."

    return (f"{correlation_desc} "
            f"Średnie stężenie PM2.5 w dniach, gdy dostępność doków gwałtownie rosła: {avg_pm25_during_increase:.2f} µg/m³.")

# Ground Truth dla pytania o strategie optymalizacji na podstawie ML i pogody (otwarte).
# To jest bardziej "podsumowanie danych", które LLM powinien zinterpretować.
def get_gt_optimization_strategies_summary(session: Session, time_interval_days: int = 7) -> str:

    start_time = datetime.now(CEST) - timedelta(days=time_interval_days)

    # Pobierz przykładowe summary_sentence z obu tabel
    bike_summary = session.query(BikesData.summary_sentence).filter(
        BikesData.timestamp >= start_time,
        BikesData.summary_sentence != ""
    ).order_by(BikesData.timestamp.desc()).first()

    bus_summary = session.query(BusesData.summary_sentence).filter(
        BusesData.timestamp >= start_time,
        BusesData.summary_sentence != ""
    ).order_by(BusesData.timestamp.desc()).first()

    bike_summary_text = bike_summary[0] if bike_summary else "Brak podsumowań dla danych rowerowych."
    bus_summary_text = bus_summary[0] if bus_summary else "Brak podsumowań dla danych autobusowych."

    # To jest miejsce, gdzie "ground truth" jest bardziej "oczekiwaną syntezą"
    general_insights = (
        "Analiza danych z ostatnich 7 dni wskazuje, że: "
        "1. Klasteryzacja stacji rowerowych może pomóc w identyfikacji obszarów o zmiennym zapotrzebowaniu na rowery. "
        "2. Prognozy opóźnień autobusów, zwłaszcza te z wysoką pewnością, są kluczowe dla zarządzania ruchem. "
        "3. Warunki pogodowe, takie jak opady i temperatura, mają wyraźny wpływ na dostępność rowerów i punktualność autobusów. "
        "4. Wysokie zanieczyszczenia mogą wpływać na preferencje transportowe. "
        "LLM powinien na podstawie tych informacji oraz konkretnych summary_sentence zasugerować strategie, np. dynamiczne przemieszczanie rowerów w zależności od prognoz pogody i klastrów, lub dostosowanie rozkładów jazdy autobusów w oparciu o przewidywane opóźnienia w specyficznych warunkach środowiskowych."
    )

    return (f"Oto podsumowania z danych z ostatnich {time_interval_days} dni, które mogą posłużyć do sformułowania strategii optymalizacji:\n\n"
            f"Podsumowanie dla rowerów: '{bike_summary_text}'\n"
            f"Podsumowanie dla autobusów: '{bus_summary_text}'\n\n"
            f"Ogólne wnioski do rozważenia przez LLM: {general_insights}")

# +-------------------------------------+
# |        SŁOWNIK PYTAŃ DO LLM         |
# |      Generowanie pytań dla LLM      |
# +-------------------------------------+

QUESTIONS_TEMPLATES = {
    "q1_bike_station_cluster_status": {
        "template": "Dla stacji rowerowej o nazwie '{station_name}', jaka jest jej predykcja klasteryzacji (cluster_id) dla ostatniego zapisu danych, i czy model uznał tę predykcję za udaną (cluster_prediction_success)? Czy w tym samym czasie występowały silne opady deszczu (precip_mm > 0)?",
        "placeholders": ["station_name"],
        "ground_truth_func": get_gt_bike_station_cluster_status,
        "time_interval_param": "time_interval_hours"
    },
    "q2_bus_cluster_delay_category": {
        "template": "Ile autobusów należało do klastra o id '{cluster_id}' w ciągu ostatnich {time_interval_hours} godzin i jaka była ich dominująca kategoria opóźnienia (delay_category_label)?",
        "placeholders": ["cluster_id"],
        "ground_truth_func": get_gt_bus_cluster_delay_category,
        "time_interval_param": "time_interval_hours"
    },
    "q3_avg_electric_bikes_success_binary": {
        "template": "Dla wszystkich predykcji bike_binary_prediction_success o wartości PRAWDA w ciągu ostatnich {time_interval_hours} godzin, jaka jest średnia liczba dostępnych rowerów elektrycznych (electric_bikes_available)?",
        "placeholders": [],
        "ground_truth_func": get_gt_avg_electric_bikes_success_binary,
        "time_interval_param": "time_interval_hours"
    },
    "q4_most_common_delay_label_success": {
        "template": "Jaka jest najczęściej występująca etykieta opóźnienia (delay_category_label) dla autobusów, dla których predykcja sukcesu opóźnienia (delay_category_prediction_success) była pomyślna w ciągu ostatnich {time_interval_hours} godzin, i jaka jest średnia wartość on_time_stop_ratio dla tych autobusów?",
        "placeholders": [],
        "ground_truth_func": get_gt_most_common_delay_label_success,
        "time_interval_param": "time_interval_hours"
    },
    "q5_weather_low_bike_availability_cluster": {
        "template": "W jakich warunkach pogodowych (weather_condition) stacje rowerowe o cluster_id równym '{cluster_id}' najczęściej miały niską dostępność rowerów (bikes_available < 5) w ciągu ostatnich {time_interval_hours} godzin?",
        "placeholders": ["cluster_id"],
        "ground_truth_func": get_gt_weather_low_bike_availability_cluster,
        "time_interval_param": "time_interval_hours"
    },
    "q6_bus_pm25_delay_correlation": {
        "template": "Czy zaobserwowano, że w przypadku autobusów linii '{bus_line_number}', dla których predykcja opóźnienia była pomyślna (delay_category_prediction_success = TRUE), średnie stężenie PM2.5 było wyższe niż w przypadku niepomyślnych predykcji w ciągu ostatnich {time_interval_hours} godzin? Podaj średnie wartości PM2.5 dla obu grup (pomyślnych i niepomyślnych predykcji).",
        "placeholders": ["bus_line_number"],
        "ground_truth_func": get_gt_bus_pm25_delay_correlation,
        "time_interval_param": "time_interval_hours"
    },
    "q7_humidity_temp_bike_prediction_impact": {
        "template": "Opisz, jak wilgotność (humidity) i temperatura (temperature) wpływają na prognozowaną liczbę dostępnych rowerów (bike_regression_prediction) dla stacji '{station_name}' w ciągu ostatnich {time_interval_hours} godzin.",
        "placeholders": ["station_name"],
        "ground_truth_func": get_gt_humidity_temp_bike_prediction_impact,
        "time_interval_param": "time_interval_hours"
    },
    "q8_visibility_very_late_buses": {
        "template": "W jakich warunkach widoczności (visibility_km) autobusy linii '{bus_line_number}' najczęściej doświadczały opóźnień sklasyfikowanych jako 'Very Late' (is_late_label = 'Very Late') w ciągu ostatnich {time_interval_hours} godzin?",
        "placeholders": ["bus_line_number"],
        "ground_truth_func": get_gt_visibility_very_late_buses,
        "time_interval_param": "time_interval_hours"
    },
    "q9_uv_bikes_bus_delay_correlation": {
        "template": "Czy istnieje trend, że w dniach o wysokim wskaźniku UV (uv_index > 7) stacje rowerowe w klastrze '{cluster_id}' mają większą liczbę dostępnych rowerów manualnych (manual_bikes_available) niż elektrycznych w ciągu ostatnich {time_interval_days} dni? I czy wpływa to na średnie opóźnienie autobusów (average_delay_seconds) w sąsiadujących obszarach (dla tego samego dnia i podobnych warunków pogodowych)?",
        "placeholders": ["cluster_id"],
        "ground_truth_func": get_gt_uv_bikes_bus_delay_correlation,
        "time_interval_param": "time_interval_days"
    },
    "q10_optimal_hours_bikes_buses": {
        "template": "W jakich godzinach dobowych (timestamp - godziny), przy niskim zachmurzeniu (cloud < 20), występuje jednocześnie największa dostępność rowerów (bikes_available) i najkrótsze średnie opóźnienia autobusów (average_delay_seconds) w ciągu ostatnich {time_interval_days} dni?",
        "placeholders": [],
        "ground_truth_func": get_gt_optimal_hours_bikes_buses,
        "time_interval_param": "time_interval_days"
    },
    "q11_top_environmental_factors_bus_delay_success": {
        "template": "Jakie są trzy główne czynniki środowiskowe (pogoda/zanieczyszczenia), które najbardziej wpływają na prognozowany sukces klasyfikacji opóźnień autobusów (delay_category_prediction_success) w ciągu ostatnich {time_interval_days} dni?",
        "placeholders": [],
        "ground_truth_func": get_gt_top_environmental_factors_bus_delay_success,
        "time_interval_param": "time_interval_days"
    },
    "q12_cluster_success_comparison_temp": {
        "template": "Porównaj sukces predykcji klasteryzacji (cluster_prediction_success) dla stacji rowerowych i linii autobusowych w ciągu ostatnich {time_interval_days} dni. Który sektor wykazał wyższy wskaźnik sukcesu i jaka była średnia temperatura (temperature) w tych pomyślnych predykcjach dla każdego sektora?",
        "placeholders": [],
        "ground_truth_func": get_gt_cluster_success_comparison_temp,
        "time_interval_param": "time_interval_days"
    },
    "q13_bike_station_needs_refill": {
        "template": "Bazując na danych o klastrach stacji rowerowych i prognozach dostępności (bike_regression_prediction), wskaż stację, która w ciągu ostatnich {time_interval_hours} godzin najprawdopodobniej wymagała uzupełnienia rowerów, biorąc pod uwagę bieżące warunki pogodowe (np. bikes_available < 5 i weather_condition jest niesprzyjający).",
        "placeholders": [],
        "ground_truth_func": get_gt_bike_station_needs_refill,
        "time_interval_param": "time_interval_hours"
    },
    "q14_bus_routes_high_delay_heavy_rain": {
        "template": "Które trasy autobusowe (bus_line_number) są najbardziej podatne na duże opóźnienia (average_delay_seconds > 300 sekund) w dniach o dużym opadzie deszczu (precip_mm > 5), zgodnie z klasyfikacjami ML w ciągu ostatnich {time_interval_days} dni?",
        "placeholders": [],
        "ground_truth_func": get_gt_bus_routes_high_delay_heavy_rain,
        "time_interval_param": "time_interval_days"
    },
    "q15_pollution_bike_rental_correlation": {
        "template": "Czy istnieje wzorzec, w którym wzrost zanieczyszczeń powietrza (np. fine_particles_pm2_5) koreluje ze spadkiem liczby wypożyczeń rowerów (wzrost docks_available)? Jakie są średnie stężenia PM2.5 w dniach, gdy docks_available rosło gwałtownie w ciągu ostatnich {time_interval_days} dni?",
        "placeholders": [],
        "ground_truth_func": get_gt_pollution_bike_rental_correlation,
        "time_interval_param": "time_interval_days"
    },
    "q16_optimization_strategies_summary": {
        "template": "Jakie strategie optymalizacji tras autobusowych lub rozmieszczenia rowerów miejskich można by zasugerować, bazując na analizie sukcesu predykcji klasteryzacji i klasyfikacji ML oraz wpływie warunków pogodowych na opóźnienia i dostępność, w oparciu o dane z ostatnich {time_interval_days} dni? Podaj ogólne wnioski.",
        "placeholders": [],
        "ground_truth_func": get_gt_optimization_strategies_summary,
        "time_interval_param": "time_interval_days"
    }
}
