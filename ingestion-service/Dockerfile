FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Dependencies
# Note the path includes the folder name because context is root
COPY ingestion-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart kafka-python pydantic redis aiokafka

# 2. Copy Shared Schemas (Critical)
COPY shared /app/shared

# 3. Copy Ingestion Source Code
COPY ingestion-service/src /app/src

# 4. Run (Host 0.0.0.0 is required for Docker networking)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]