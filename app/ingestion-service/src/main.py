import os
import json
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from aiokafka import AIOKafkaProducer

# Import ONLY the new schemas
from .schemas import VideoIngestRequest, TaskDispatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")

app = FastAPI(title="KCE Ingestion Service")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")

# Map "Pipeline Names" to "Kafka Topics"
TOPIC_MAP = {
    "face": "face-tasks",
    "body": "body-tasks",
    "audio": "audio-tasks"
}

producer = None


# --- LIFECYCLE ---
@app.on_event("startup")
async def startup():
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    logger.info("Kafka Producer Connected")


@app.on_event("shutdown")
async def shutdown():
    if producer:
        await producer.stop()

# --- HELPER ---


async def dispatch_to_kafka(topic: str, payload: dict):
    try:
        msg = json.dumps(payload).encode('utf-8')
        await producer.send_and_wait(topic, msg)
    except Exception as e:
        logger.error(f"Kafka Publish Error ({topic}): {e}")
        raise HTTPException(status_code=500, detail="Event Bus Failure")

# --- ENDPOINTS ---


@app.post("/internal/ingest/video")
async def ingest_video(request: VideoIngestRequest):
    """
    Fan-out Strategy:
    Gateway sends 1 request -> We fire N Kafka Events (Face, Body, Audio)
    """
    triggered = []

    # Convert Pydantic model to Dict ONCE
    payload = request.dict()

    for pipeline in request.pipelines:
        if pipeline in TOPIC_MAP:
            target_topic = TOPIC_MAP[pipeline]
            await dispatch_to_kafka(target_topic, payload)
            triggered.append(pipeline)
        else:
            logger.warning(f"Skipping unknown pipeline: {pipeline}")

    if not triggered:
        raise HTTPException(
            status_code=400, detail="No valid pipelines requested")

    return {
        "status": "dispatched",
        "task_id": request.task_id,
        "pipelines_triggered": triggered
    }


@app.post("/internal/dispatch/{worker_type}")
async def dispatch_specific(worker_type: str, task: TaskDispatch):
    """
    Direct Injection for testing or specific re-runs.
    """
    if worker_type not in TOPIC_MAP:
        raise HTTPException(
            status_code=404, detail=f"Worker '{worker_type}' not configured")

    target_topic = TOPIC_MAP[worker_type]

    # Forward the task_id and metadata to the worker
    await dispatch_to_kafka(target_topic, task.dict())

    return {"status": "dispatched", "target_topic": target_topic}


@app.get("/health")
def health():
    return {"status": "active", "role": "orchestrator"}
