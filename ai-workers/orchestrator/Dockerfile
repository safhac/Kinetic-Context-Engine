FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir kafka-python pydantic

# Copy shared first
COPY shared /app/shared

# If your file is at ai-workers/orchestrator/main.py, copy it like this:
COPY ai-workers/orchestrator /app/orchestrator

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Update the command to match the location
CMD ["python", "orchestrator/main.py"]