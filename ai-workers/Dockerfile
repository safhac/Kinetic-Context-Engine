FROM python:3.9-slim

# Install system dependencies
# Added ffmpeg, libsm6, libxext6 for Video Encoding
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsndfile1 \
    libasound2 \
    libglib2.0-0 \
    libpulse0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./ai-workers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./shared /app/shared
COPY ./ai-workers /app/ai-workers

CMD ["python", "ai-workers/body_worker/tasks.py"]