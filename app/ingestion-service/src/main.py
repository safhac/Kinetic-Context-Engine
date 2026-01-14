import os
import json
import logging
import redis  # <--- ADDED IMPORT
from fastapi import FastAPI, HTTPException
from aiokafka import AIOKafkaProducer
from .schemas import VideoIngestRequest, TaskDispatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")

app = FastAPI(title="KCE Ingestion Service")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")  # <--- ADDED CONFIG

TOPIC_MAP = {
    "face": "face-tasks",
    "body": "body-tasks",
    "audio": "audio-tasks"
}

# --- REDIS CLIENT ---
# Initialize connection to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=6379,
                           db=0, decode_responses=True)

producer = None


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


async def dispatch_to_kafka(topic: str, payload: dict):
    try:
        msg = json.dumps(payload).encode('utf-8')
        await producer.send_and_wait(topic, msg)
        logger.info(f"Dispatched to {topic}")
    except Exception as e:
        logger.error(f"Kafka Publish Error ({topic}): {e}")


@app.post("/internal/ingest/video")
async def ingest_video(request: VideoIngestRequest):
    payload = request.dict()
    triggered = []

    # Calculate how many valid workers we are about to trigger
    active_pipelines = [p for p in request.pipelines if p in TOPIC_MAP]
    count = len(active_pipelines)

    # FIX 1: Use 'request' instead of 'task'
    # FIX 2: Ensure we save the count dynamically, not hardcoded to 3
    if count > 0:
        redis_client.set(f"path:{request.task_id}", request.file_path)
        redis_client.set(f"pending:{request.task_id}", count)

    for pipeline in request.pipelines:
        if pipeline in TOPIC_MAP:
            target_topic = TOPIC_MAP[pipeline]
            await dispatch_to_kafka(target_topic, payload)
            triggered.append(pipeline)

    return {"status": "dispatched", "pipelines": triggered}


@app.get("/health")
def health():
    return {"status": "active"}
