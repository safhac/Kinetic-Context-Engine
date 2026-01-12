import os
import shutil
import uuid
import json
import logging
import httpx  # Async HTTP client
import asyncio
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiokafka import AIOKafkaConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")

app = FastAPI(title="KCE Public Gateway")

# Config
INGESTION_URL = os.getenv("INGESTION_SERVICE_URL",
                          "http://ingestion-service:8001")
UPLOAD_DIR = "/app/media/uploads"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
TOPIC_RESULTS = "interpreted_context"  # Gateway ONLY listens to results

# CORS & Static
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# SSE State
active_connections: Dict[str, List[asyncio.Queue]] = {}


@app.on_event("startup")
async def startup():
    # Start background listener for Results (for SSE)
    asyncio.create_task(result_listener())


async def result_listener():
    consumer = AIOKafkaConsumer(TOPIC_RESULTS, bootstrap_servers=KAFKA_BROKER)
    await consumer.start()
    try:
        async for msg in consumer:
            data = json.loads(msg.value)
            task_id = data.get("task_id")
            if task_id in active_connections:
                for q in active_connections[task_id]:
                    await q.put(data)
    finally:
        await consumer.stop()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), context: str = Form("general")):
    """
    1. Save File (IO)
    2. Ping Ingestion Service (Network)
    """
    task_id = str(uuid.uuid4())
    filename = f"{task_id}.{file.filename.split('.')[-1]}"
    file_path = f"{UPLOAD_DIR}/{filename}"

    # 1. Dumb IO
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Handoff to Ingestion Service
    payload = {
        "task_id": task_id,
        "file_path": file_path,
        "context": context,
        "original_name": file.filename
    }

    async with httpx.AsyncClient() as client:
        try:
            # We fire and forget (await response, but don't wait for processing)
            resp = await client.post(f"{INGESTION_URL}/internal/ingest", json=payload)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to contact Ingestion Service: {e}")
            raise HTTPException(
                status_code=503, detail="Processing Service Unavailable")

    return {"task_id": task_id, "status": "queued"}


@app.get("/stream/{task_id}")
async def stream(request: Request, task_id: str):
    """Standard SSE implementation"""
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
