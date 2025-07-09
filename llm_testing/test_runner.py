from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import List, Any
import random
import time

# --- Importy połączenia się i funkcji łączących się z PostGreSQL i innych ---
from shared.db_dto import BikesData, BusesData, LLMTestResult, CEST
from shared.anything_wrapper import query_workspace_llm
from questions import QUESTIONS_TEMPLATES
from shared.db_conn import SessionLocal
from shared.db_utils import save_log

# --- Ustawienia podstawowe ---
ANYTHING_LLM_WORKSPACE_ID = "project" 
TEST_CYCLE_INTERVAL_HOURS = 6 
ID_SAMPLE_LIMIT = 5 

# +-------------------------------------+
# |         FUNKCJE POMOCNICZE          |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

# Pobiera unikalne ID z określonej kolumny w ramach ostatniego interwału czasowego. Zwraca losową próbkę tych ID.
def get_recent_ids(session: Session, dto_class: Any, id_column: Any, limit: int, time_interval_hours: int) -> List[Any]:

    start_time = datetime.now(CEST) - timedelta(hours=time_interval_hours)
    
    all_recent_ids = session.execute(
        select(id_column)
        .filter(dto_class.timestamp >= start_time)
        .distinct()
    ).scalars().all()

    # Jeśli jest mniej ID niż limit, bierzemy wszystkie dostępne
    if len(all_recent_ids) <= limit:
        return all_recent_ids
    
    # W przeciwnym razie, zwracamy losową próbkę
    return random.sample(all_recent_ids, limit)

# Zapisuje wynik pojedynczego testu do bazy danych.
def save_test_result(session: Session, test_run_timestamp: datetime, question_key: str, 
                     question_text: str, ground_truth: str, llm_response: str, 
                     response_time_ms: float, llm_error: bool, llm_error_message: str = None):

    try:
        result = LLMTestResult(
            timestamp=test_run_timestamp,
            question_key=question_key,
            question_text=question_text,
            ground_truth=ground_truth,
            llm_response=llm_response,
            response_time_ms=response_time_ms,
            llm_error=llm_error,
            llm_error_message=llm_error_message
        )
        session.add(result)
        session.commit()
        print(f"✅ Zapisano wynik testu dla pytania '{question_key}'.")
    except Exception as e:
        session.rollback()
        print(f"❌ BŁĄD podczas zapisu wyniku testu dla pytania '{question_key}': {e}")

# +-------------------------------------+
# |     GŁÓWNA FUNKCJA WYKONUJĄCA       |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

# Główna funkcja uruchamiająca cykliczne testy LLM.
def run_cyclic_tests_llm():

    print("🚀 Uruchamiam cykliczny runner testów LLM...")
    save_log("llm_tester", "info", "LLM_Tester został uruchomiony")

    while True:
        test_start_time = datetime.now(CEST) 
        print(f"\n--- Rozpoczynam nowy cykl testowy: {test_start_time.isoformat()} ---")
        
        db_session = SessionLocal()
        try:
            # 1. Pobierz reprezentatywne ID z najnowszych danych
            recent_bike_stations = get_recent_ids(db_session, BikesData, BikesData.name, ID_SAMPLE_LIMIT, TEST_CYCLE_INTERVAL_HOURS)
            recent_bus_lines = get_recent_ids(db_session, BusesData, BusesData.bus_line_number, ID_SAMPLE_LIMIT, TEST_CYCLE_INTERVAL_HOURS)
            
            recent_bike_cluster_ids = get_recent_ids(db_session, BikesData, BikesData.cluster_id, ID_SAMPLE_LIMIT, TEST_CYCLE_INTERVAL_HOURS)
            recent_bus_cluster_ids = get_recent_ids(db_session, BusesData, BusesData.cluster_id, ID_SAMPLE_LIMIT, TEST_CYCLE_INTERVAL_HOURS)
            
            all_recent_cluster_ids = list(set(recent_bike_cluster_ids + recent_bus_cluster_ids))
            if not all_recent_cluster_ids and (any("cluster_id" in q_info["placeholders"] for q_info in QUESTIONS_TEMPLATES.values())):
                print("⚠️ Brak dostępnych ID klastrów w ostatnich danych. Niektóre pytania mogą być pominięte.")
                save_log("llm_tester", "warning", "Brak dostępnych ID klastrów z ostatnich danych. Niektóre pytania mogą być pominięte.")
            
            # 2. Iteruj po szablonach pytań i generuj testy
            for q_key, q_info in QUESTIONS_TEMPLATES.items():
                question_template = q_info["template"]
                ground_truth_func = q_info["ground_truth_func"]
                time_param_name = q_info.get("time_interval_param", "time_interval_hours")

                # Ustawienie interwału czasowego dla funkcji GT
                current_time_interval = TEST_CYCLE_INTERVAL_HOURS
                if time_param_name == "time_interval_days":
                    current_time_interval = TEST_CYCLE_INTERVAL_HOURS / 24
                    if current_time_interval < 1:
                        current_time_interval = 7

                # Lista parametrów do iteracji dla tego pytania
                params_to_iterate = [{}]

                if "station_name" in q_info["placeholders"]:
                    if recent_bike_stations:
                        params_to_iterate = [{"station_name": name} for name in recent_bike_stations]
                    else:
                        print(f"   Pominięto pytanie '{q_key}': Brak dostępnych nazw stacji rowerowych.")
                        continue
                elif "bus_line_number" in q_info["placeholders"]:
                    if recent_bus_lines:
                        params_to_iterate = [{"bus_line_number": line} for line in recent_bus_lines]
                    else:
                        print(f"   Pominięto pytanie '{q_key}': Brak dostępnych numerów linii autobusowych.")
                        continue
                elif "cluster_id" in q_info["placeholders"]:
                    
                    # Sprawdź, czy pytanie dotyczy klastrów rowerowych czy autobusowych
                    if all_recent_cluster_ids:
                        params_to_iterate = [{"cluster_id": cid} for cid in all_recent_cluster_ids]
                    else:
                        print(f"   Pominięto pytanie '{q_key}': Brak dostępnych ID klastrów.")
                        continue
                
                for params in params_to_iterate:
                    # Dodaj parametr czasowy do parametrów funkcji GT
                    gt_params = {**params, time_param_name: int(current_time_interval)}

                    try:
                        # Generuj pytanie do LLM
                        question_text = question_template.format(**params, **{time_param_name: int(current_time_interval)})
                        print(f"   Generuję Ground Truth dla pytania '{q_key}' z parametrami: {params}...")
                        save_log("llm_tester", "info", f" Generuję Ground Truth dla pytania '{q_key}' z parametrami: {params}...")
                        ground_truth = ground_truth_func(db_session, **gt_params)
                        
                        print(f"   Wysyłam zapytanie do LLM: {question_text[:100]}...")
                        save_log("llm_tester", "info", f" Wysyłam zapytanie do LLM: {question_text[:100]}...")
                        llm_start_time = time.perf_counter()
                        llm_response_data = query_workspace_llm(ANYTHING_LLM_WORKSPACE_ID, question_text)
                        llm_end_time = time.perf_counter()
                        response_time_ms = (llm_end_time - llm_start_time) * 1000

                        llm_response_text = llm_response_data.get("textResponse", "")
                        llm_error = llm_response_data.get("error", False)
                        llm_error_message = llm_response_data.get("message", None)

                        save_test_result(
                            db_session,
                            test_start_time,
                            q_key,
                            question_text,
                            ground_truth,
                            llm_response_text,
                            response_time_ms,
                            llm_error,
                            llm_error_message
                        )

                    except Exception as e:
                        print(f"❌ Wystąpił błąd podczas przetwarzania pytania '{q_key}' z parametrami {params}: {e}")
                        save_log("llm_tester", "error", f"Wystąpił błąd podczas przetwarzania pytania '{q_key}' z parametrami {params}: {e}")
                        
                        # Zapisz błąd, nawet jeśli LLM nie odpowiedział
                        save_test_result(
                            db_session,
                            test_start_time,
                            q_key,
                            question_text,
                            ground_truth,
                            llm_response_text,
                            response_time_ms,
                            True,
                            f"Błąd wewnętrzny runnera: {e}"
                        )
        except Exception as e:
            print(f"❌ Krytyczny błąd w cyklu testowym: {e}")
            save_log("llm_tester", "error", f"Krytyczny błąd w cyklu testowym: {e}")
        finally:
            db_session.close()
            print(f"--- Cykl testowy zakończony. Następny za {TEST_CYCLE_INTERVAL_HOURS} godzin. ---")
            save_log("llm_tester", "info", f"Cykl testowy zakończony. Następny za {TEST_CYCLE_INTERVAL_HOURS} godzin.")
            time.sleep(TEST_CYCLE_INTERVAL_HOURS * 3600)

# +-------------------------------------+
# |     GŁÓWNA FUNKCJA WKYONUJĄCA       |
# |       Proces odpytywanie LLM        |
# +-------------------------------------+

if __name__ == "__main__":
    # Uruchom runner testów
    run_cyclic_tests_llm()
