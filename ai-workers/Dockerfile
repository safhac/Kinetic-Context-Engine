FROM python:3.9-slim

WORKDIR /app

COPY ./ai-workers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./shared /app/shared
COPY ./ai-workers /app/ai-workers

# Entrypoint defined in docker-compose
CMD ["python", "ai-workers/body-worker/main.py"]