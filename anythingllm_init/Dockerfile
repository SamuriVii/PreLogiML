FROM python:3.12-slim

WORKDIR /app

# Skopiuj requirements
COPY anythingllm_init/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj shared modules
COPY shared /app/shared

# Skopiuj skrypt inicjatora
COPY anythingllm_init/anythingllm_init.py .

# Ustaw permissions
RUN chmod +x anythingllm_init.py

CMD ["python", "anythingllm_init.py"]