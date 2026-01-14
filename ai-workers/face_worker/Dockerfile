FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ai-workers/face_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared /app/shared
COPY ai-workers/face_worker /app/ai-workers/face_worker
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/ai-workers/face_worker
CMD ["python", "ai-workers/face_worker/main.py"]