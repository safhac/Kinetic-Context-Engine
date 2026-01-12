import os
import shutil
import asyncio
import json
import logging
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from schemas import VideoUploadTask, ProcessingEvent

# --- CONFIG ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BROKER", "kafka:29092")
TOPIC_TELEMETRY = "raw_telemetry"
TOPIC_NOTIFICATIONS = "interpreted_context"  # Where completion events come from
UPLOAD_DIR = "/app/media/uploads"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")


app = FastAPI(title="KCE Ingestion Gateway")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount Static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- GLOBAL STATE (Broadcaster Pattern) ---
# Maps task_id -> List[asyncio.Queue]
# When Kafka says "Task X done", we put data in all Queues for Task X.
active_connections: Dict[str, List[asyncio.Queue]] = {}

# --- KAFKA LIFECYCLE ---
producer = None
consumer = None


@app.post("/process/body")
async def process_body(task: ImageTask):
    # producer.send('body-tasks', json.dumps(task.dict()).encode('utf-8'))
    return {"status": "queued", "queue": "body-tasks", "task_id": task.task_id}


@app.post("/process/face")
async def process_face(task: ImageTask):
    # producer.send('face-tasks', json.dumps(task.dict()).encode('utf-8'))
    return {"status": "queued", "queue": "face-tasks", "task_id": task.task_id}


@app.get("/health")
def health():
    return {"status": "ok"}
