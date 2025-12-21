from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import logging

from .schemas import TelemetryPayload, VideoMetadata

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KCE Ingestion Gate")

# --- Configuration ---
# Uses environment variables for configuration (12-Factor App)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "raw-telemetry")

# Placeholder for Producer (Initialized on startup)
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

@app.post("/ingest")
async def ingest_telemetry(payload: TelemetryPayload):
    """
    High-throughput endpoint receiving raw video metadata and telemetry.
    """
    if not producer:
        raise HTTPException(status_code=503, detail="Data bus unavailable")

    try:
        # Push to Fault-Tolerant Data Bus 
        future = producer.send(TOPIC_NAME, key=payload.session_id.encode('utf-8'), value=payload.dict())
        return {"status": "queued", "partition": future.get().partition}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")

@app.get("/health")
def health_check():
    return {"status": "active", "service": "ingestion-gate"}