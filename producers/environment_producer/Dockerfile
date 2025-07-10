FROM python:3.12-slim 

WORKDIR /app

COPY shared /app/shared
COPY producers/environment_producer /app

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python3", "app.py"]