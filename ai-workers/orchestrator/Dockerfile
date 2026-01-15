FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --no-cache-dir kafka-python pydantic

# 1. Copy the shared contract
COPY shared /app/shared

# 2. Copy the unified logic (Main + Model + Rules)
COPY ai-workers/orchestrator /app/orchestrator

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "orchestrator/main.py"]