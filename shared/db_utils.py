from shared.db_dto import LogsEntry, EnvironmentEntry, BikesData, BusesData, ModelStatus, CEST
from shared.db_conn import SessionLocal
from datetime import datetime, timezone
from sqlalchemy import func, select
from typing import Optional
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

# +-------------------------------------+
# |     WERYFIKACJA MODELU - CUSTER     |
# |      Funkcje Czytajace Rekordy      |
# +-------------------------------------+

# Pobiera status modelu z bazy danych.
def get_model_status(model_name: str) -> Optional[ModelStatus]:
    db = SessionLocal()
    try:
        model_status_entry = db.execute(
            select(ModelStatus).filter_by(model_name=model_name)
        ).scalar_one_or_none()
        return model_status_entry
    except Exception as e:
        save_log(f"db_utils_get_status_{model_name}", "error", f"Błąd podczas pobierania statusu modelu '{model_name}': {e}")
        return None
    finally:
        db.close()

# Aktualizuje flagę is_new_model_available dla danego modelu.
def update_model_new_available_flag(model_name: str, status: bool):
    db = SessionLocal()
    try:
        model_status_entry = db.execute(
            select(ModelStatus).filter_by(model_name=model_name)
        ).scalar_one_or_none()
        
        if model_status_entry:
            model_status_entry.is_new_model_available = status
            model_status_entry.last_updated = datetime.now(CEST)
            db.commit()
            save_log(f"db_utils_update_flag_{model_name}", "info", f"Ustawiono flagę is_new_model_available na {status} dla modelu '{model_name}'.")
        else:
            save_log(f"db_utils_update_flag_{model_name}", "warning", f"Nie znaleziono wpisu dla modelu '{model_name}' w celu aktualizacji flagi.")
    except Exception as e:
        db.rollback()
        save_log(f"db_utils_update_flag_{model_name}", "error", f"Błąd podczas aktualizacji flagi dla modelu '{model_name}': {e}")
    finally:
        db.close()
