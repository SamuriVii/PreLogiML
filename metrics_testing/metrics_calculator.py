from nltk.translate.bleu_score import sentence_bleu
from shared.db_conn import SessionLocal
from shared.db_dto import LLMTestResult
from shared.db_utils import save_log
from rouge_score import rouge_scorer
from sqlalchemy.orm import Session
from sqlalchemy import select
import nltk
import time

# --- Upewnij się, że masz pobrane dane NLTK potrzebne do tokenizacji ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    nltk.download('punkt')

# +-------------------------------------+
# |         FUNKCJE POMOCNICZE          |
# |        Moduł metryk jakości         |
# +-------------------------------------+

# Oblicza metryki BLEU i ROUGE
def calculate_metrics(reference: str, hypothesis: str) -> dict:

    if not reference or not hypothesis:
        return {"bleu": None, "rouge": None, "error": "Brak tekstu referencyjnego lub hipotezy."}

    # Tokenizacja dla BLEU. BLEU działa lepiej na tokenach, a nie na całym stringu
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())

    # BLEU score - dodano warunek, aby uniknąć błędu Empty text w przypadku pustych tokenów
    if len(reference_tokens) > 0 and len(hypothesis_tokens) > 0:
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    else:
        bleu_score = 0.0

    # ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    rouge_l_fscore = scores['rougeL'].fmeasure

    return {"bleu": bleu_score, "rouge": rouge_l_fscore, "error": None}

# +-------------------------------------+
# |           FUNKCJA GŁÓWNA            |
# |        Moduł metryk jakości         |
# +-------------------------------------+

def run_metrics_calculation_cycle():
    print("🚀 Uruchamiam cykl obliczania metryk BLEU i ROUGE...")
    save_log("metrics_calculator", "info", "Metrics Calculator został uruchomiony.")

    session = SessionLocal()
    try:
        # 1. Pobierz testy bez obliczonych metryk i upewnij się, że kolumny bleu_score i rouge_score istnieją w LLMTestResult.
        tests_to_process = session.execute(
            select(LLMTestResult)
            .filter((LLMTestResult.bleu_score.is_(None)) | (LLMTestResult.rouge_score.is_(None)))
            .filter(LLMTestResult.llm_error == False)
        ).scalars().all()

        if not tests_to_process:
            print("Brak nowych wyników do obliczenia metryk.")
            save_log("metrics_calculator", "info", "Brak nowych wyników do obliczenia metryk.")
            return

        print(f"Znaleziono {len(tests_to_process)} wyników do przetworzenia.")
        for test_result in tests_to_process:
            # Pomiń, jeśli Ground Truth lub odpowiedź LLM są puste lub nie są stringami
            if not isinstance(test_result.ground_truth, str) or not test_result.ground_truth.strip():
                print(f"Pominięto test '{test_result.question_key}' (ID: {test_result.id}): Pusty Ground Truth.")
                test_result.bleu_score = None
                test_result.rouge_score = None
                continue
            if not isinstance(test_result.llm_response, str) or not test_result.llm_response.strip():
                print(f"Pominięto test '{test_result.question_key}' (ID: {test_result.id}): Pusta odpowiedź LLM.")
                test_result.bleu_score = None
                test_result.rouge_score = None
                continue

            metrics = calculate_metrics(test_result.ground_truth, test_result.llm_response)
            
            if metrics["error"]:
                print(f"Błąd obliczania metryk dla '{test_result.question_key}' (ID: {test_result.id}): {metrics['error']}")
                save_log("metrics_calculator", "error", f"Błąd obliczania metryk dla '{test_result.question_key}': {metrics['error']}")
                # Możesz tu zdecydować, czy zapisywać None, czy konkretną wartość błędu
                test_result.bleu_score = None
                test_result.rouge_score = None
            else:
                test_result.bleu_score = metrics["bleu"]
                test_result.rouge_score = metrics["rouge"]
                print(f"Obliczono metryki dla '{test_result.question_key}' (ID: {test_result.id}): BLEU={metrics['bleu']:.4f}, ROUGE-L={metrics['rouge']:.4f}")
            
            session.add(test_result)

        session.commit()
        print("Cykl obliczania metryk zakończony pomyślnie.")
        save_log("metrics_calculator", "info", "Cykl obliczania metryk zakończony pomyślnie.")

    except Exception as e:
        session.rollback()
        print(f"❌ Krytyczny błąd w cyklu obliczania metryk: {e}")
        save_log("metrics_calculator", "error", f"Krytyczny błąd w cyklu obliczania metryk: {e}")
    finally:
        session.close()

if __name__ == "__main__":

    while True:
        run_metrics_calculation_cycle()
        time.sleep(3600)
