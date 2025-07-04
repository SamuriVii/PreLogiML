from typing import Dict, Any
import requests
import time

# --- Importy połączenia się i funkcji łączących się z PostGreSQL ---
from .db_utils import save_log

# --- Ustawienia autoryzacji dla AnythingLLM ---
API_URL = "http://anythingllm:3001"
API_KEY = "R2R6MPA-64Q4MZS-G1MWH0G-PEABVK9"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}





def get_workspaces() -> Dict[Any, Any]:
    url = f"{API_URL}/api/v1/workspaces"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        print(response)
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"BŁĄD: {e}")
        return {
            "error": True,
            "message": f"Błąd połączenia: {str(e)}",
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }

