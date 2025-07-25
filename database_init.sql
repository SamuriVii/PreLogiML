-- creating "logs" table
CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    service VARCHAR(255) NOT NULL,
    information_type VARCHAR(50) NOT NULL,
    event TEXT NOT NULL
);

-- creating "enviroment_data" table
CREATE TABLE IF NOT EXISTS environment_data (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    city VARCHAR(100) NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    temperature NUMERIC(5,2),
    feelslike NUMERIC(5,2),
    humidity INT,
    wind_kph NUMERIC(5,2),
    precip_mm NUMERIC(5,2),
    cloud INT,
    visibility_km NUMERIC(5,2),
    uv_index NUMERIC(4,2),
    is_day VARCHAR(10),
    condition TEXT,
    pm2_5 NUMERIC(6,2),
    pm10 NUMERIC(6,2),
    co NUMERIC(8,2),
    no2 NUMERIC(6,2),
    o3 NUMERIC(6,2),
    so2 NUMERIC(6,2)
);

-- creating "bikes_data" table
CREATE TABLE IF NOT EXISTS bikes_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    station_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    bikes_available INTEGER NOT NULL,
    capacity INTEGER,
    docks_available INTEGER,
    manual_bikes_available INTEGER,
    electric_bikes_available INTEGER,
    temperature REAL,
    feelslike REAL,
    humidity INTEGER,
    wind_kph REAL,
    precip_mm REAL,
    cloud INTEGER,
    visibility_km REAL,
    uv_index REAL,
    daylight VARCHAR(50) NOT NULL,
    weather_condition VARCHAR(255),
    fine_particles_pm2_5 REAL,
    coarse_particles_pm10 REAL,
    carbon_monoxide_ppb REAL,
    nitrogen_dioxide_ppb REAL,
    ozone_ppb REAL,
    sulfur_dioxide_ppb REAL,
    cluster_id INTEGER,
    cluster_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    bike_binary_prediction INTEGER,
    bike_binary_probabilities TEXT,
    bike_binary_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    bike_binary_label VARCHAR(255),
    bike_multiclass_prediction INTEGER,
    bike_multiclass_probabilities TEXT,
    bike_multiclass_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    bike_multiclass_label VARCHAR(255),
    bike_regression_prediction REAL,
    bike_regression_prediction_original REAL,
    bike_regression_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    summary_sentence TEXT NOT NULL DEFAULT ''
);

-- creating "buses_data" table
CREATE TABLE IF NOT EXISTS buses_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    vehicle_id VARCHAR(50) NOT NULL,
    bus_line_number VARCHAR(50) NOT NULL,
    route_destination VARCHAR(255) NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    speed REAL,
    direction REAL,
    trip_identifier VARCHAR(100),
    route_identifier VARCHAR(100),
    stops_count INTEGER,
    average_delay_seconds REAL,
    maximum_delay_seconds REAL,
    minimum_delay_seconds REAL,
    delay_variance_value REAL,
    delay_standard_deviation REAL,
    delay_range_seconds REAL,
    stops_on_time_count INTEGER,
    stops_arrived_early_count INTEGER,
    stops_arrived_late_count INTEGER,
    delay_trend VARCHAR(50),
    delay_consistency_score REAL,
    on_time_stop_ratio REAL,
    avg_positive_delay_seconds REAL,
    avg_negative_delay_seconds REAL,
    temperature REAL,
    feelslike REAL,
    humidity INTEGER,
    wind_kph REAL,
    precip_mm REAL,
    cloud INTEGER,
    visibility_km REAL,
    uv_index REAL,
    daylight VARCHAR(50) NOT NULL,
    weather_condition VARCHAR(255),
    fine_particles_pm2_5 REAL,
    coarse_particles_pm10 REAL,
    carbon_monoxide_ppb REAL,
    nitrogen_dioxide_ppb REAL,
    ozone_ppb REAL,
    sulfur_dioxide_ppb REAL,
    cluster_id INTEGER,
    cluster_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    is_late_prediction INTEGER,
    is_late_probabilities TEXT,
    is_late_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    is_late_label VARCHAR(255),
    delay_category_prediction INTEGER,
    delay_category_probabilities TEXT,
    delay_category_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    delay_category_label VARCHAR(255),
    average_delay_seconds_prediction REAL,
    average_delay_seconds_prediction_original REAL,
    average_delay_seconds_prediction_success BOOLEAN NOT NULL DEFAULT FALSE,
    summary_sentence TEXT NOT NULL DEFAULT ''
);

-- creating "llm_test_results" table
CREATE TABLE IF NOT EXISTS llm_test_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    question_key VARCHAR(255) NOT NULL,
    question_text TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    llm_response TEXT,
    response_time_ms FLOAT,
    llm_error BOOLEAN NOT NULL DEFAULT FALSE,
    llm_error_message TEXT,
    bleu_score FLOAT,
    rouge_score FLOAT
);

-- creating "model_status" table
CREATE TABLE IF NOT EXISTS model_status (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,
    is_new_model_available BOOLEAN NOT NULL DEFAULT FALSE,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quality_metric FLOAT,
    version INTEGER NOT NULL DEFAULT 1
);

-- inserting data into "model_status" table
INSERT INTO model_status (model_name, is_new_model_available, quality_metric, version) VALUES
('bikes_kmeans', FALSE, 0.0, 1),
('bikes_binary_classifier', FALSE, 0.0, 1),
('bikes_multiclass_classifier', FALSE, 0.0, 1),
('bikes_regression_predictor', FALSE, 0.0, 1),
('buses_kmeans', FALSE, 0.0, 1),
('bus_binary_classifier', FALSE, 0.0, 1),
('bus_multiclass_classifier', FALSE, 0.0, 1),
('bus_regression_predictor', FALSE, 0.0, 1)
ON CONFLICT (model_name) DO NOTHING;
