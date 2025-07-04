from shared.db_dto import LogsEntry, EnvironmentEntry, BikesData, BusesData
from shared.db_conn import SessionLocal
from datetime import datetime, timezone
from sqlalchemy import func
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

# +-------------------------------------+
# |         ZAPISYWANIE DANYCH          |
# |      Funkcje Zapisujące Rekordy     |
# +-------------------------------------+

# Funkcja zapisująca rekord danych (dane rowerowe)
def save_bike_data_to_base(final_sql_data: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(final_sql_data)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bike_record = BikesData(**data_to_save)
        db.add(bike_record)
        db.commit()
        save_log("subscriber_bikes", "info", "Zapisano dane.")
    except Exception as e:
        db.rollback()
        save_log("subscriber_bikes", "error", f"Wystąpił błąd podczas zapisu danych: {e}.")

# Funkcja zapisująca rekord (dane autobusowe)
def save_bus_data_to_base(final_sql_data: dict):
    db = SessionLocal()
    try:
        data_to_save = deepcopy(final_sql_data)
        data_to_save["timestamp"] = datetime.fromtimestamp(data_to_save.get("timestamp", time.time()), tz=timezone.utc)

        bus_record = BusesData(**data_to_save)
        db.add(bus_record)
        db.commit()
        save_log("subscriber_buses", "info", "Zapisano dane.")
    except Exception as e:
        db.rollback()
        save_log("subscriber_buses", "error", f"Wystąpił błąd podczas zapisu danych: {e}.")
