FROM python:3.9-slim

# libgl1 and libglib2.0-0 replace the deprecated mesa-glx
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The context is the project root, so paths are relative to that
COPY ai-workers/face_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
COPY ai-workers/face_worker /app/ai-workers/face_worker

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/ai-workers/face_worker

CMD ["python", "ai-workers/face_worker/main.py"]