FROM python:3.9-slim

# Install ffmpeg for ffprobe/metadata support
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# We only need kafka and pydantic for the Orchestrator
RUN pip install --no-cache-dir kafka-python pydantic

# COPY logic
# Note: We copy the src into /app/src and the shared schemas into /app/shared
COPY shared /app/shared
COPY ai-workers/orchestrator/src /app/src

ENV PYTHONUNBUFFERED=1
# Ensure PYTHONPATH allows 'import shared.schemas' and 'import src.main'
ENV PYTHONPATH=/app

CMD ["python", "src/main.py"]