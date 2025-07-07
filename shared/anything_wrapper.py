from typing import Dict, Any
import requests
import json

# --- Ustawienia autoryzacji dla AnythingLLM i ChromaVDB ---
API_URL = "http://anythingllm:3001"
API_KEY = "B2YJHSE-BBPMT0V-M74MPZ6-KAPRV2N"

CHROMA_DB_HOST = "chroma"
CHROMA_DB_PORT = 8000

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "accept": "application/json"
}

# +-------------------------------------+
# |       FUNKCJA ODPYTUJĄCE LLM        |
# |      Komunikacja z AnythingLLM      |
# +-------------------------------------+

# Wysyła zapytanie do modelu LLM w ramach określonego workspace'u AnythingLLM.
def query_workspace_llm(workspace_id: str, message: str) -> Dict[Any, Any]:
    endpoint = f"/api/v1/workspace/{workspace_id}/chat"
    payload = {
        "message": message,
        "mode": "query"
    }

    print(f"Wysyłanie zapytania do workspace'u '{workspace_id}': '{message}'...")
    
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status() # Wywoła wyjątek dla kodów statusu 4xx/5xx
        print(f"Sukces: {response.status_code} dla {endpoint}")
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"BŁĄD podczas wysyłania zapytania do LLM: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" - Status: {e.response.status_code}"
            try:
                error_message += f" - Odpowiedź: {e.response.json()}"
            except json.JSONDecodeError:
                error_message += f" - Odpowiedź (tekst): {e.response.text}"
        
        print(error_message)
        return {
            "error": True,
            "message": error_message,
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }

# +-------------------------------------+
# |    FUNKCJE ZAPISUJĄCE EMBEDDINGI    |
# |      Komunikacja z AnythingLLM      |
# +-------------------------------------+

# Dodaje pre-wyliczony embedding danych rowerowych wraz z tekstem i metadanymi do AnythingLLM.
def add_raw_bike_text_to_anythingllm(workspace_slug: str, raw_text_content: str, metadata: Dict[str, Any]) -> Dict[Any, Any]:
    endpoint = "/api/v1/document/raw-text"
    url = f"{API_URL}{endpoint}"

    # ANYTHINGLLM wymaga klucza 'title' w metadanych dla tego endpointu
    if "title" not in metadata:
        # Możesz np. użyć combination of station_id and timestamp for a unique title
        metadata["title"] = f"Bike Data - {metadata.get('station_id', 'Unknown')}-{metadata.get('timestamp', 'No Timestamp')}"

    payload = {
        "textContent": raw_text_content,
        "addToWorkspaces": workspace_slug,
        "metadata": metadata
    }

    print(f"\n🚀 Wysyłanie surowego tekstu do AnythingLLM (Workspace: {workspace_slug})...")
    print(f"Payload do wysłania (bez full metadata): textContent='{raw_text_content[:100]}...', metadata keys={list(metadata.keys())}")

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status() # Wywoła wyjątek dla kodów statusu 4xx/5xx
        print(f"✅ Sukces: {response.status_code} dla {endpoint}")
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"❌ BŁĄD podczas wysyłania surowego tekstu do AnythingLLM: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" - Status: {e.response.status_code}"
            try:
                error_message += f" - Odpowiedź: {e.response.json()}"
            except json.JSONDecodeError:
                error_message += f" - Odpowiedź (tekst): {e.response.text}"
        
        print(error_message)
        return {
            "error": True,
            "message": error_message,
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }

# Dodaje pre-wyliczony embedding danych autobusowych wraz z tekstem i metadanymi do AnythingLLM.
def add_raw_bus_text_to_anythingllm(workspace_slug: str, raw_text_content: str, metadata: Dict[str, Any]) -> Dict[Any, Any]:

    endpoint = "/api/v1/document/raw-text"
    url = f"{API_URL}{endpoint}"

    # ANYTHINGLLM wymaga klucza 'title' w metadanych dla tego endpointu
    if "title" not in metadata:
        # Tworzenie unikalnego tytułu na podstawie vehicle_id i timestamp
        metadata["title"] = f"Bus Data - {metadata.get('vehicle_id', 'Unknown')}-{metadata.get('timestamp', 'No Timestamp')}"

    payload = {
        "textContent": raw_text_content,
        "addToWorkspaces": workspace_slug,
        "metadata": metadata
    }

    print(f"\n🚀 Wysyłanie surowego tekstu dla autobusu do AnythingLLM (Workspace: {workspace_slug})...")
    print(f"Payload do wysłania (bez full metadata): textContent='{raw_text_content[:100]}...', metadata keys={list(metadata.keys())}")

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status() # Wywoła wyjątek dla kodów statusu 4xx/5xx
        print(f"✅ Sukces: {response.status_code} dla {endpoint}")
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"❌ BŁĄD podczas wysyłania surowego tekstu dla autobusu do AnythingLLM: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" - Status: {e.response.status_code}"
            try:
                error_message += f" - Odpowiedź: {e.response.json()}"
            except json.JSONDecodeError:
                error_message += f" - Odpowiedź (tekst): {e.response.text}"
        
        print(error_message)
        return {
            "error": True,
            "message": error_message,
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }
