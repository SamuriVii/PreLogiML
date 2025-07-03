from shared.db_dto import LogsEntry, EnvironmentEntry, BikeCluster, BusesCluster, BikeClass, BusesClass
from shared.db_conn import SessionLocal
from sqlalchemy import func
from datetime import datetime, timezone
from copy import deepcopy
import time

# Funkcja zapisująca logi do bazy
def save_log(service: str, info_type: str, event: str):
    db = SessionLocal()
    try:
        log = LogsEntry(service=service, information_type=info_type, event=event)
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"❌ Błąd zapisu logu: {e}")
        db.rollback()
    finally:
        db.close()

# Funkcja dopasowująca dane środowiskowe z najbliższej możliwej daty i godziny
def get_closest_environment(session, target_ts: datetime):
    return (
        session.query(EnvironmentEntry).order_by(func.abs(func.extract('epoch', EnvironmentEntry.timestamp) - target_ts.timestamp())).first()
    )

# Funkcja zapisująca rekord danych uczących model klasteryzacji do bazy (dane rowerowe)
def save_bike_cluster_record(enriched: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(enriched)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bike_record = BikeCluster(**data_to_save)
        db.add(bike_record)
        db.commit()
        print("🚲 BikeCluster record saved.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving BikeCluster record: {e}")

# Funkcja zapisująca rekord danych uczących model klasteryzacji do bazy (dane autobusowe)
def save_bus_cluster_record(enriched: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(enriched)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bus_record = BusesCluster(**data_to_save)
        db.add(bus_record)
        db.commit()
        print("🚌 BusesCluster record saved.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving BusesCluster record: {e}")

# Funkcja zapisująca rekord danych uczących model klasyfikacji do bazy (dane rowerowe)
def save_bike_class_record(enriched: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(enriched)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bike_record = BikeClass(**data_to_save)
        db.add(bike_record)
        db.commit()
        print("🚲 BikeClass record saved.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving BikeClass record: {e}")

# Funkcja zapisująca rekord danych uczących model klasyfikacji do bazy (dane autobusowe)
def save_bus_class_record(enriched: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(enriched)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bus_record = BusesClass(**data_to_save)
        db.add(bus_record)
        db.commit()
        print("🚌 BusesClass record saved.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving BusesClass record: {e}")
















