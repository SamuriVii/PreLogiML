from datetime import datetime, timedelta, timezone
from sqlalchemy import Column, Integer, String, TIMESTAMP, Double, Numeric, Text, Float, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()
CEST = timezone(timedelta(hours=2))

class LogsEntry(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(CEST))
    service = Column(String, unique=True, nullable=False)
    information_type = Column(String, nullable=False)
    event = Column(String, nullable=False)

class EnvironmentEntry(Base):
    __tablename__ = "environment_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(CEST))
    city = Column(String, nullable=False)
    lat = Column(Double)
    lon = Column(Double)
    temperature = Column(Numeric)
    feelslike = Column(Numeric)
    humidity = Column(Integer)
    wind_kph = Column(Numeric)
    precip_mm = Column(Numeric)
    cloud = Column(Integer)
    visibility_km = Column(Numeric)
    uv_index = Column(Numeric)
    is_day = Column(String)
    condition = Column(Text)
    pm2_5 = Column(Numeric)
    pm10 = Column(Numeric)
    co = Column(Numeric)
    no2 = Column(Numeric)
    o3 = Column(Numeric)
    so2 = Column(Numeric)

class BikesData(Base):
    __tablename__ = 'bikes_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(CEST))
    
    station_id = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    bikes_available = Column(Integer, nullable=False)
    
    capacity = Column(Integer, nullable=True)
    docks_available = Column(Integer, nullable=True)
    manual_bikes_available = Column(Integer, nullable=True)
    electric_bikes_available = Column(Integer, nullable=True)

    temperature = Column(Float, nullable=True)
    feelslike = Column(Float, nullable=True)
    humidity = Column(Integer, nullable=True)
    wind_kph = Column(Float, nullable=True)
    precip_mm = Column(Float, nullable=True)
    cloud = Column(Integer, nullable=True)
    visibility_km = Column(Float, nullable=True)
    uv_index = Column(Float, nullable=True)
    daylight = Column(String(50), nullable=False)
    weather_condition = Column(String(255), nullable=True)

    fine_particles_pm2_5 = Column(Float, nullable=True)
    coarse_particles_pm10 = Column(Float, nullable=True)
    carbon_monoxide_ppb = Column(Float, nullable=True)
    nitrogen_dioxide_ppb = Column(Float, nullable=True)
    ozone_ppb = Column(Float, nullable=True)
    sulfur_dioxide_ppb = Column(Float, nullable=True)

    cluster_id = Column(Integer, nullable=True)
    cluster_prediction_success = Column(Boolean, nullable=False, default=False)

    bike_binary_prediction = Column(Integer, nullable=True)
    bike_binary_probabilities = Column(Text, nullable=True)
    bike_binary_prediction_success = Column(Boolean, nullable=False, default=False)
    bike_binary_label = Column(String(255), nullable=True)

    bike_multiclass_prediction = Column(Integer, nullable=True)
    bike_multiclass_probabilities = Column(Text, nullable=True)
    bike_multiclass_prediction_success = Column(Boolean, nullable=False, default=False)
    bike_multiclass_label = Column(String(255), nullable=True)

    bike_regression_prediction = Column(Float, nullable=True)
    bike_regression_prediction_original = Column(Float, nullable=True)
    bike_regression_prediction_success = Column(Boolean, nullable=False, default=False)

    summary_sentence = Column(Text, nullable=False, default="")

class BusesData(Base):
    __tablename__ = 'buses_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(CEST))
    
    vehicle_id = Column(String(50), nullable=False)
    bus_line_number = Column(String(50), nullable=False)
    route_destination = Column(String(255), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    speed = Column(Float, nullable=True)
    direction = Column(Float, nullable=True)
    trip_identifier = Column(String(100), nullable=True)
    route_identifier = Column(String(100), nullable=True)
    stops_count = Column(Integer, nullable=True)

    average_delay_seconds = Column(Float, nullable=True)
    maximum_delay_seconds = Column(Float, nullable=True)
    minimum_delay_seconds = Column(Float, nullable=True)
    delay_variance_value = Column(Float, nullable=True)
    delay_standard_deviation = Column(Float, nullable=True)
    delay_range_seconds = Column(Float, nullable=True)
    stops_on_time_count = Column(Integer, nullable=True)
    stops_arrived_early_count = Column(Integer, nullable=True)
    stops_arrived_late_count = Column(Integer, nullable=True)
    delay_trend = Column(String(50), nullable=True)
    delay_consistency_score = Column(Float, nullable=True)
    on_time_stop_ratio = Column(Float, nullable=True)
    avg_positive_delay_seconds = Column(Float, nullable=True)
    avg_negative_delay_seconds = Column(Float, nullable=True)

    temperature = Column(Float, nullable=True)
    feelslike = Column(Float, nullable=True)
    humidity = Column(Integer, nullable=True)
    wind_kph = Column(Float, nullable=True)
    precip_mm = Column(Float, nullable=True)
    cloud = Column(Integer, nullable=True)
    visibility_km = Column(Float, nullable=True)
    uv_index = Column(Float, nullable=True)
    daylight = Column(String(50), nullable=False)
    weather_condition = Column(String(255), nullable=True)

    fine_particles_pm2_5 = Column(Float, nullable=True)
    coarse_particles_pm10 = Column(Float, nullable=True)
    carbon_monoxide_ppb = Column(Float, nullable=True)
    nitrogen_dioxide_ppb = Column(Float, nullable=True)
    ozone_ppb = Column(Float, nullable=True)
    sulfur_dioxide_ppb = Column(Float, nullable=True)

    cluster_id = Column(Integer, nullable=True)
    cluster_prediction_success = Column(Boolean, nullable=False, default=False)

    is_late_prediction = Column(Integer, nullable=True)
    is_late_probabilities = Column(Text, nullable=True)
    is_late_prediction_success = Column(Boolean, nullable=False, default=False)
    is_late_label = Column(String(255), nullable=True)

    delay_category_prediction = Column(Integer, nullable=True)
    delay_category_probabilities = Column(Text, nullable=True)
    delay_category_prediction_success = Column(Boolean, nullable=False, default=False)
    delay_category_label = Column(String(255), nullable=True)

    average_delay_seconds_prediction = Column(Float, nullable=True)
    average_delay_seconds_prediction_original = Column(Float, nullable=True)
    average_delay_seconds_prediction_success = Column(Boolean, nullable=False, default=False)

    summary_sentence = Column(Text, nullable=False, default="")

class LLMTestResult(Base):
    __tablename__ = 'llm_test_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(CEST)) 
    question_key = Column(String(255), nullable=False)
    question_text = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    llm_response = Column(Text, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    llm_error = Column(Boolean, nullable=False, default=False)
    llm_error_message = Column(Text, nullable=True)
    bleu_score = Column(Float, nullable=True)
    rouge_score = Column(Float, nullable=True)
