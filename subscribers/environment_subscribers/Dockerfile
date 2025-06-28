FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY subscribers/environment_subscribers /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
