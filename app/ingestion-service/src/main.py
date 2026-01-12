import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")

app = FastAPI(title="KCE Ingestion Service")

# Config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
# Topics
TOPIC_FACE = "face-tasks"
TOPIC_BODY = "body-tasks"
TOPIC_AUDIO = "audio-tasks"

producer = None


@app.on_event("startup")
async def startup():
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()


@app.on_event("shutdown")
async def shutdown():
    if producer:
        await producer.stop()

# --- DATA MODELS ---


class VideoIngest(BaseModel):
    task_id: str
    file_path: str
    context: str
    original_name: str


class TaskDispatch(BaseModel):
    task_id: str
    metadata: dict = {}

# --- ENDPOINTS ---


@app.post("/internal/ingest/video")
async def ingest_video(payload: VideoIngest):
    """Main Pipeline: Fan-out to ALL workers"""
    msg = json.dumps(payload.dict()).encode('utf-8')
    try:
        # Start all 3 workers for a full video analysis
        await producer.send_and_wait(TOPIC_FACE, msg)
        await producer.send_and_wait(TOPIC_BODY, msg)
        await producer.send_and_wait(TOPIC_AUDIO, msg)
        return {"status": "ok", "pipeline": "full_video"}
    except Exception as e:
        logger.error(f"Kafka Error: {e}")
        raise HTTPException(status_code=500, detail="Kafka Failure")


@app.post("/internal/dispatch/body")
async def dispatch_body(payload: TaskDispatch):
    """Specific Pipeline: Body Only"""
    await producer.send_and_wait(TOPIC_BODY, json.dumps(payload.dict()).encode('utf-8'))
    return {"status": "ok", "target": "body_worker"}


@app.post("/internal/dispatch/face")
async def dispatch_face(payload: TaskDispatch):
    """Specific Pipeline: Face Only"""
    await producer.send_and_wait(TOPIC_FACE, json.dumps(payload.dict()).encode('utf-8'))
    return {"status": "ok", "target": "face_worker"}


@app.post("/internal/dispatch/audio")
async def dispatch_audio(payload: TaskDispatch):
    """Specific Pipeline: Audio Only"""
    await producer.send_and_wait(TOPIC_AUDIO, json.dumps(payload.dict()).encode('utf-8'))
    return {"status": "ok", "target": "audio_worker"}


@app.get("/health")
def health():
    return {"status": "active"}
