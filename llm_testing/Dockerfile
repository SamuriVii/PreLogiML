FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY llm_testing /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "test_runner.py"]