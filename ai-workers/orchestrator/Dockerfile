FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir kafka-python pydantic

# 1. Copy the contract
COPY shared /app/shared
# 2. Copy ONLY the logic we need from the messy context-engine
COPY app/context-engine/src/deception_model.py /app/app/context-engine/src/
COPY app/context-engine/src/rules.json /app/app/context-engine/src/
# 3. Copy the orchestrator script
COPY ai-workers/orchestrator /app/orchestrator

ENV PYTHONPATH=/app
CMD ["python", "orchestrator/main.py"]