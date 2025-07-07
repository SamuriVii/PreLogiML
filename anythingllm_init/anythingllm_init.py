from shared.anything_wrapper import API_URL, API_KEY, CHROMA_DB_HOST, CHROMA_DB_PORT
from shared.db_utils import save_log
import chromadb
import requests
import time

# --- Opóźnienie startu ---
print("Kontener startuje")
time.sleep(120)

# --- Ustawienia workspace'u ---
WORKSPACE_NAME = "Project"

# +-------------------------------------+
# |         FUNKCJE POMOCNICZE        |
# |       Inicjacja AnythingLLM       |
# +-------------------------------------+

# Sprawdza status połączenia z AnythingLLM API.
def check_anythingllm_api_status() -> bool:
    print(f"\nSprawdzanie statusu AnythingLLM API pod adresem: {API_URL}/api/v1/workspaces...")
    try:
        # Używamy prostego endpointu do sprawdzenia dostępności
        response = requests.get(f"{API_URL}/api/v1/workspaces", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=10)
        response.raise_for_status()
        print(f"✅ AnythingLLM API jest dostępne! Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ BŁĄD: Nie można połączyć się z AnythingLLM API: {e}")
        return False

# Sprawdza status połączenia z ChromaDB.
def check_chromadb_status() -> bool:
    print(f"\nSprawdzanie statusu ChromaDB pod adresem: http://{CHROMA_DB_HOST}:{CHROMA_DB_PORT}...")
    try:
        client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
        # Prosta operacja, która wymaga połączenia z bazą
        client.heartbeat()
        print("✅ ChromaDB jest dostępne!")
        return True
    except Exception as e:
        print(f"❌ BŁĄD: Nie można połączyć się z ChromaDB: {e}")
        return False

# +-------------------------------------+
# |          FUNKCJA GŁÓWNA           |
# |        Inicjacja AnythingLLM        |
# +-------------------------------------+

def main():
    print("🚀 Rozpoczynanie kreatora workspace'u AnythingLLM...")

    # Sprawdzenie statusu AnythingLLM API
    if not check_anythingllm_api_status():
        print("🛑 AnythingLLM API nie jest dostępne. Zakończenie.")
        save_log("anythingllm_init", "error", "Brak połączenia z AnythingLLM")
        exit(1)
    save_log("anythingllm_init", "info", "Połączono z AnythingLLM")

    # Sprawdzenie statusu ChromaDB
    if not check_chromadb_status():
        print("🛑 ChromaDB nie jest dostępne. Zakończenie.")
        save_log("anythingllm_init", "error", "Brak połączenia z ChromaDB")
        exit(1)
    save_log("anythingllm_init", "info", "Połączono z ChromaDB")

    print("\n🎉 Kreator workspace'u zakończył działanie.")

# +-------------------------------------+
# |         FUNKCJA WYKONAWCZA        |
# |        Inicjacja AnythingLLM        |
# +-------------------------------------+

if __name__ == "__main__":
    main()
