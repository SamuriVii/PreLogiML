FROM python:3.12-slim

WORKDIR /app

# Skopiuj requirements
COPY llm_tester/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj shared modules
COPY shared /app/shared

# Skopiuj skrypt testera
COPY llm_tester/llm_tester.py .

# Ustaw permissions
RUN chmod +x llm_tester.py

CMD ["python", "llm_tester.py"]