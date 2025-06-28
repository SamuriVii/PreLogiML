from shared.db_dto import LogsEntry, EnvironmentEntry
from shared.db_conn import SessionLocal
from sqlalchemy import func
import datetime

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


















