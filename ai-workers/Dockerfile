FROM python:3.9-slim

# Install system dependencies
# libgl1: The modern equivalent for libGL.so.1
# libglib2.0-0: Required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./ai-workers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./shared /app/shared
COPY ./ai-workers /app/ai-workers

# Note: This CMD is overridden by docker-compose, so it doesn't matter much.
# But for correctness, update the path if you renamed the folder to body_worker
CMD ["python", "ai-workers/body_worker/tasks.py"]