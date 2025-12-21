from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import shutil
import uuid
import logging
from celery import Celery

# Import schemas (Ensure schemas.py exists in the src folder or shared path)
from .schemas import TelemetryPayload 

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KCE Ingestion Gate")

# --- Configuration ---
# 1. Kafka Config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "raw-telemetry")

# 2. Redis/Celery Config (New)
CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
celery_client = Celery('ingestion', broker=CELERY_BROKER)

# 3. Shared Storage Config (New)
UPLOAD_DIR = "/app/media/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Placeholder for Kafka Producer
producer = None

@app.on_event("startup")
async def startup_event():
    global producer
    try:
        from kafka import KafkaProducer
        import json
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Connected to Kafka at {KAFKA_BROKER}")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")

# --- Existing Telemetry Endpoint ---
@app.post("/ingest")
async def ingest_telemetry(payload: TelemetryPayload):
    """
    High-throughput endpoint receiving raw video metadata and telemetry.
    """
    if not producer:
        raise HTTPException(status_code=503, detail="Data bus unavailable")

    try:
        # Push to Kafka
        future = producer.send(TOPIC_NAME, key=payload.session_id.encode('utf-8'), value=payload.dict())
        return {"status": "queued", "partition": future.get().partition}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")

# --- NEW: Video Upload Endpoint ---
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives a video, saves it to Shared Volume, and triggers AI Worker.
    """
    try:
        # 1. Generate ID and Path
        task_id = str(uuid.uuid4())
        filename = f"{task_id}.mp4"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # 2. Save file to disk (Shared Volume)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Trigger Celery Task
        # Must match name in ai-workers/body-worker/tasks.py
        celery_client.send_task(
            'process_body_video', 
            args=[file_path, task_id],
            task_id=task_id
        )

        logger.info(f"Video {task_id} queued for processing.")
        return {
            "status": "queued", 
            "task_id": task_id, 
            "message": "Video accepted. Processing started in background."
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Clean up partial file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "service": "ingestion-gate"}