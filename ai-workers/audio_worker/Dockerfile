FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libpulse0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ai-workers/audio_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
COPY ai-workers/audio_worker /app/ai-workers/audio_worker

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/ai-workers/audio_worker

CMD ["python", "ai-workers/audio_worker/main.py"]