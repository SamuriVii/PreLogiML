# 🤖 System AI dla Inteligentnego Miasta

Ten projekt to kompleksowe rozwiązanie wykorzystujące sztuczną inteligencję do monitorowania miejskiej infrastruktury. System łączy w sobie możliwości dużych modeli językowych (LLM), baz danych wektorowych, monitoringu przepływu danych oraz tradycyjnych modeli uczenia maszynowego, aby dostarczać inteligentne predykcje danych rowerowych (Mevo) oraz autobusowych (ZKM).

---

## ℹ️ Podstawowe Informacje

Ten projekt integruje szereg technologii, które współdziałają w ekosystemie Docker Compose. Poniżej znajdziesz kluczowe adresy, pod którymi możesz monitorować i zarządzać poszczególnymi komponentami systemu po jego uruchomieniu:

* **PGAdmin4:** [http://localhost:8080](http://localhost:8080) – Interfejs graficzny do zarządzania bazami danych PostgreSQL. Tutaj znajdziesz wyniki predykcji i inne dane operacyjne.
* **ChromaDB:** [http://chroma:8000](http://chroma:8000) – Twoja baza danych wektorowych, wykorzystywana przez AnythingLLM do przechowywania i wyszukiwania osadzeń.
* **AnythingLLM:** [http://localhost:3001](http://localhost:3001) – Platforma do budowania aplikacji LLM z bazami wiedzy. Służy do interakcji z modelem językowym w kontekście danych z systemu.
* **Node-RED:** [http://localhost:1880](http://localhost:1880) – Środowisko do wizualnego programowania przepływów danych. Tutaj możesz monitorować częstotliwość uruchamiania i status poszczególnych procesów w systemie.

---

## 🚀 Instrukcja Uruchamiania Systemu

Aby poprawnie uruchomić cały ekosystem, wykonaj poniższe kroki:

1.  **Sklonuj Repozytorium:**
    ```bash
    git clone <URL_TWOJEGO_REPOZYTORIUM>
    cd <NAZWA_TWOJEGO_REPOZYTORIUM>
    ```

2.  **Uruchom Docker Compose:**
    Upewnij się, że masz zainstalowanego Dockera i Docker Compose (zalecane środowisko to WSL z Dockerem na systemach Windows). Następnie przejdź do katalogu głównego projektu i uruchom wszystkie usługi:
    ```bash
    docker compose up -d
    ```
    Poczekaj chwilę, aż wszystkie usługi zostaną pobrane i uruchomione. Proces ten może potrwać kilka minut przy pierwszym uruchomieniu.

3.  **Weryfikacja Działania:**
    Po uruchomieniu kontenerów możesz sprawdzić ich status:
    ```bash
    docker compose ps
    ```
    Wszystkie usługi powinny być w stanie `running`.

---

## 🔑 Konfiguracja AnythingLLM (Po Uruchomieniu)

Po uruchomieniu systemu, należy skonfigurować AnythingLLM, aby połączyć go z wybranym modelem językowym i bazą danych wektorowych.

1.  **Otwórz AnythingLLM:** Przejdź do [http://localhost:3001](http://localhost:3001) w swojej przeglądarce.
2.  **Ustawienia Dostawcy LLM:**
    * Przejdź do **Ustawień** (Settings) w interfejsie AnythingLLM.
    * Wybierz sekcję **Providers**.
    * Jako dostawcę modelu LLM wybierz **OpenAI** (lub innego preferowanego dostawcę).
    * **Generowanie Klucza API OpenAI:** Jeśli używasz OpenAI, wygeneruj swój klucz API na platformie OpenAI: [https://platform.openai.com/settings/organization/api-keys](https://platform.openai.com/settings/organization/api-keys).
    * Wprowadź swój wygenerowany **klucz API** (np. `sk-proj-LXKa8BtDxP36K5oXXX`) w odpowiednie pole w AnythingLLM.
    * Wybierz preferowany model LLM (np. `gpt-4o`, `gpt-3.5-turbo`).
3.  **Ustawienia Bazy Wektorowej (Vector Database):**
    * W sekcji **Vector Database** wybierz **ChromaDB**.
    * W polu adresu wprowadź: `http://chroma:8000`.
    * Zapisz zmiany.

Gotowe! AnythingLLM jest teraz połączone z Twoim modelem LLM i bazą danych wektorowych.

---

## 📈 Monitorowanie i Wyniki

* **Node-RED:** Na stronie [http://localhost:1880](http://localhost:1880) możesz obserwować przepływy danych i częstotliwość uruchamiania poszczególnych procesów. W przyszłości planujemy dodać szczegółowe opisy odpowiedzialności każdego z procesów, a także **zrzuty ekranu** (możesz wstawiać je używając składni `![Opis zdjęcia](ścieżka/do/zdjęcia.png)`) dla lepszej wizualizacji.
* **PGAdmin4:** W PGAdmin4, po połączeniu się z bazą danych, możesz przeglądać tabelę `llm_test_results`, która przechowuje wyniki predykcji i inne istotne dane generowane przez system.

---

## 🛠️ Dalszy Rozwój

Obecnie README zawiera podstawowe informacje i instrukcje. W kolejnych etapach rozwoju projektu planujemy rozbudować dokumentację o:

* Szczegółową strukturę bazy danych (schematy tabel, opisy kolumn).
* Dokumentację dotyczącą funkcji i klas w kodzie.
* Przykładowe zastosowania i scenariusze użycia systemu.
* Więcej zrzutów ekranu i wizualizacji działania.
