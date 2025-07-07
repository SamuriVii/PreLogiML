FROM python:3.12-slim

WORKDIR /app

COPY metrics_testing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
COPY metrics_testing /app

ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p ${NLTK_DATA}
RUN python -c "import nltk; nltk.download('all', download_dir='${NLTK_DATA}', quiet=True)"

CMD ["python", "metrics_calculator.py"]