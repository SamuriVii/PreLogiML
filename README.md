POŁĄCZENIE SIĘ Z PGADMIN4: http://localhost:8080

PAMIĘTAJ, ŻEBY OPÓŹNIĆ START Z 60 NA 300 SEKUND DLA PRODUCENTÓW DANYCH AUTOBUSOWYCH ORAZ ROWEROWYCH - TAK SAMO ODBIORCÓW
DLACZEGO? MUSIMY POCZEKAĆ NA PIERWSZY REKORD W BAZIE DANYCH Z DANYMI ŚRODOWISKOWYMI

KAŻDY Z SUBÓW MA PO PROSTU W SOBIE 4 FUNKCJE:
    1. Przekształcenie danych - przygotowanie pod ML -- GOTOWE
    2. Przygotowanie zdania z danych "dict" dla emmbeddingów - może ręcznie, może tam właśnie klasyfikację dodatkową zrobić lub zamiast obecnej
    3. Klasteryzacja
    4. Klasyfikacja
    5. ML na embeddingi


POŁĄCZENIE DO ANYTHINGLLM TERMINALEM - DZIAŁA

PS C:\Users\szymo> Invoke-WebRequest -Uri "http://localhost:3001/api/v1/workspaces" `
>>   -Headers @{
>>     "Authorization" = "Bearer VYZ40NS-HJY4G18-MAFD8EB-YF06RMA"
>>     "accept" = "application/json"
>>   } `
>>   -Method GET