FROM python:3.9-slim

# Install system dependencies for Audio (libsndfile for librosa, ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Requirements
COPY ai-workers/audio_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Shared Code & Worker Code
COPY shared /app/shared
COPY ai-workers/audio_worker /app/ai-workers/audio_worker

# Set Python Path
ENV PYTHONPATH=/app:/app/ai-workers/audio_worker

# Run
CMD ["python", "ai-workers/audio_worker/main.py"]