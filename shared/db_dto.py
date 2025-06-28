from datetime import datetime, timedelta, timezone
from sqlalchemy import Column, Integer, String, TIMESTAMP, Double, Numeric, Text
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


