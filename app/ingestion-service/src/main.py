import os
import json
import logging
import redis  # <--- ADDED IMPORT
from fastapi import FastAPI, HTTPException
from aiokafka import AIOKafkaProducer
from .schemas import VideoIngestRequest, TaskDispatch, VideoProfile

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

    # 1. PROFILE THE VIDEO (New Step)
    try:
        profile = VideoProfile.from_file(request.task_id, request.file_path)
        # Send profile to the new 'video_profiles' topic for the Orchestrator
        await dispatch_to_kafka("video_profiles", profile.dict())
        logger.info(
            f"📊 Video Profile Sent: Vertical={profile.is_vertical}, Audio={profile.has_audio}")
    except Exception as e:
        logger.error(f"❌ Profiling Failed for {request.task_id}: {e}")
        # Continue anyway, but Orchestrator will use defaults

    # 2. REDIS STATE (Your existing logic)
    active_pipelines = [p for p in request.pipelines if p in TOPIC_MAP]
    count = len(active_pipelines)

    if count > 0:
        redis_client.set(f"path:{request.task_id}", request.file_path)
        redis_client.set(f"pending:{request.task_id}", count)

    # 3. DISPATCH TO WORKERS
    for pipeline in request.pipelines:
        if pipeline in TOPIC_MAP:
            target_topic = TOPIC_MAP[pipeline]
            await dispatch_to_kafka(target_topic, payload)
            triggered.append(pipeline)

    return {"status": "dispatched", "pipelines": triggered, "profile": profile.dict()}


@app.get("/health")
def health():
    return {"status": "active"}
