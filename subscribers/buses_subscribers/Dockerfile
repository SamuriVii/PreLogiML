FROM python:3.12-slim

WORKDIR /app

COPY subscribers/buses_subscribers /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
