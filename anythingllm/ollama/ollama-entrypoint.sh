#!/bin/bash

echo "🚀 Startowanie Ollama server w tle..."
ollama serve &

# Czekaj aż serwer będzie gotowy
echo "⏳ Oczekiwanie na gotowość serwera Ollama..."
until curl --silent --fail http://localhost:11434; do
  sleep 1
done

# Pobierz model, jeśli nie istnieje
if ! curl -s http://localhost:11434/api/tags | grep -q '"name":"llama3:8b"'; then
  echo "⬇️ Pobieranie modelu llama3:8b..."
  ollama pull llama3:8b
else
  echo "✅ Model llama3:8b już istnieje."
fi

echo "🎉 Ollama gotowa do użycia!"

# Zablokuj kontener na procesie serwera
wait