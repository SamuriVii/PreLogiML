FROM ollama/ollama:0.9.3

# Instaluj curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Kopiuj skrypt entrypoint
COPY ./ollama-entrypoint.sh /usr/local/bin/ollama-entrypoint.sh

# Ustaw odpowiednie uprawnienia i format Unix
RUN chmod +x /usr/local/bin/ollama-entrypoint.sh && \
    sed -i 's/\r$//' /usr/local/bin/ollama-entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/ollama-entrypoint.sh"]