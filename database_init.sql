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

    -- Pogoda
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

    -- Jakość powietrza
    pm2_5 NUMERIC(6,2),
    pm10 NUMERIC(6,2),
    co NUMERIC(8,2),
    no2 NUMERIC(6,2),
    o3 NUMERIC(6,2),
    so2 NUMERIC(6,2)
);



