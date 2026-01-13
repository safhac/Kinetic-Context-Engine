import os
import shutil
import uuid
import json
import logging
import httpx
import asyncio
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")

app = FastAPI(title="KCE Public Gateway")

# Config
INGESTION_URL = os.getenv("INGESTION_SERVICE_URL",
                          "http://ingestion-service:8001")
UPLOAD_DIR = "/app/media/uploads"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
TOPIC_RESULTS = "interpreted_context"

# CORS & Static
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# SSE State
active_connections: Dict[str, List[asyncio.Queue]] = {}

# --- SSE LISTENER ---


@app.on_event("startup")
async def startup():
    asyncio.create_task(result_listener())


async def result_listener():
    consumer = AIOKafkaConsumer(TOPIC_RESULTS, bootstrap_servers=KAFKA_BROKER)

    # --- RETRY LOGIC START ---
    max_retries = 10
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Connecting to Kafka (Attempt {attempt+1}/{max_retries})...")
            await consumer.start()
            logger.info("✅ Successfully connected to Kafka.")
            break
        except KafkaConnectionError:
            if attempt == max_retries - 1:
                logger.error("❌ Max retries reached. Kafka is unavailable.")
                raise
            logger.warning("Kafka not ready. Retrying in 5 seconds...")
            await asyncio.sleep(5)
    # --- RETRY LOGIC END ---

    try:
        async for msg in consumer:
            data = json.loads(msg.value)
            task_id = data.get("task_id")
            if task_id in active_connections:
                for q in active_connections[task_id]:
                    await q.put(data)
    finally:
        await consumer.stop()
# --- UPLOAD ENDPOINT ---


@app.post("/ingest/upload")
async def upload_file(file: UploadFile = File(...), context: str = Form("general")):
    task_id = str(uuid.uuid4())
    filename = f"{task_id}.{file.filename.split('.')[-1]}"
    file_path = f"{UPLOAD_DIR}/{filename}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    payload = {"task_id": task_id, "file_path": file_path,
               "context": context, "original_name": file.filename}

    # Forward to Ingestion Service
    async with httpx.AsyncClient() as client:
        await client.post(f"{INGESTION_URL}/internal/ingest/video", json=payload)

    return {"task_id": task_id, "status": "queued"}

# --- RESTORED: SPECIFIC ENDPOINTS ---


class TaskModel(BaseModel):
    task_id: str
    metadata: dict = {}


@app.post("/process/body")
async def process_body(task: TaskModel):
    async with httpx.AsyncClient() as client:
        await client.post(f"{INGESTION_URL}/internal/dispatch/body", json=task.dict())
    return {"status": "dispatched", "type": "body"}


@app.post("/process/face")
async def process_face(task: TaskModel):
    async with httpx.AsyncClient() as client:
        await client.post(f"{INGESTION_URL}/internal/dispatch/face", json=task.dict())
    return {"status": "dispatched", "type": "face"}


@app.post("/process/audio")
async def process_audio(task: TaskModel):
    async with httpx.AsyncClient() as client:
        await client.post(f"{INGESTION_URL}/internal/dispatch/audio", json=task.dict())
    return {"status": "dispatched", "type": "audio"}

# --- SSE STREAM ---


@app.get("/ingest/stream/{task_id}")
async def stream(request: Request, task_id: str):
    async def generator():
        q = asyncio.Queue()
        if task_id not in active_connections:
            active_connections[task_id] = []
        active_connections[task_id].append(q)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ["completed", "failed"]:
                        break
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            active_connections[task_id].remove(q)
    return StreamingResponse(generator(), media_type="text/event-stream")
