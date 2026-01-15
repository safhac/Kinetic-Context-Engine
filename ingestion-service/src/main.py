import os
import asyncio
import json
import logging
import uuid
import shutil
import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from .schemas import VideoProfile  # Ensure this exists in your schemas.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")

app = FastAPI(title="KCE Ingestion Service (Unified)")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
UPLOAD_DIR = "/app/media/uploads"

# --- CORS (Added so Frontend can talk directly) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REDIS & KAFKA CLIENTS ---
redis_client = redis.Redis(host=REDIS_HOST, port=6379,
                           db=0, decode_responses=True)
producer = None


@app.on_event("startup")
async def startup():
    global producer
    # Create Upload Dir if missing
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Start Kafka
    # Retry Loop for Kafka
    max_retries = 10
    for i in range(max_retries):
        try:
            logger.info(f"Connecting to Kafka ({i+1}/{max_retries})...")
            producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
            await producer.start()
            logger.info("✅ Unified Ingestor Connected to Kafka")
            return
        except (KafkaConnectionError, OSError) as e:
            logger.warning(f"Kafka not ready yet: {e}")
            await asyncio.sleep(5)

    raise Exception("Could not connect to Kafka after retries")


@app.on_event("shutdown")
async def shutdown():
    if producer:
        await producer.stop()

# --- HELPER: Dispatch ---


async def dispatch_to_kafka(topic: str, payload: dict):
    try:
        msg = json.dumps(payload).encode('utf-8')
        await producer.send_and_wait(topic, msg)
    except Exception as e:
        logger.error(f"Kafka Error ({topic}): {e}")

# --- THE HYBRID ENDPOINT ---
# This combines Gateway's "Upload" with Ingestor's "Dispatch"


@app.post("/ingest/upload")
async def upload_video(file: UploadFile = File(...), context: str = Form("general")):
    try:
        # 1. GATEWAY LOGIC: Handle the physical file
        task_id = str(uuid.uuid4())
        filename = f"{task_id}.{file.filename.split('.')[-1]}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"💾 File Saved: {file_path}")

        # 2. INGESTOR LOGIC: Profile the video
        try:
            profile = VideoProfile.from_file(task_id, file_path)
            await dispatch_to_kafka("video_profiles", profile.dict())
        except Exception as e:
            logger.warning(f"Profiling skipped for {task_id}: {e}")
            # Fallback profile if needed, or let orchestration handle it

        # 3. DISPATCH LOGIC: Send to workers
        payload = {
            "task_id": task_id,
            "file_path": file_path,
            "context": context
        }

        # Fire and forget to all 3 workers
        for topic in ["face-tasks", "body-tasks", "audio-tasks"]:
            await dispatch_to_kafka(topic, payload)

        # 4. REDIS LOGIC: Set state for tracking
        redis_client.set(f"path:{task_id}", file_path)
        redis_client.set(f"pending:{task_id}", 3)  # Expecting 3 results

        return {
            "task_id": task_id,
            "status": "processing",
            "file_path": file_path
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "active", "mode": "unified"}
