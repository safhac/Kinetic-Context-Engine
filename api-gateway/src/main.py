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


@app.on_event("startup")
async def startup_event():
    global producer, consumer
    # 1. Setup Producer (Command Side)
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()

    # 2. Setup Consumer (Query Side - Background Listener)
    consumer = AIOKafkaConsumer(
        TOPIC_NOTIFICATIONS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="gateway_broadcaster"
    )
    await consumer.start()
    asyncio.create_task(broadcast_listener())


@app.on_event("shutdown")
async def shutdown_event():
    if producer:
        await producer.stop()
    if consumer:
        await consumer.stop()


# --- BACKGROUND WORKER ---
async def broadcast_listener():
    """Reads from Kafka Notification Topic and dispatches to SSE clients."""
    try:
        async for msg in consumer:
            data = json.loads(msg.value)
            task_id = data.get("task_id") or data.get("session_id")

            if task_id in active_connections:
                logger.info(f"Broadcasting update for {task_id}")
                for q in active_connections[task_id]:
                    await q.put(data)
    except Exception as e:
        logger.error(f"Broadcaster Error: {e}")

# --- COMMAND: UPLOAD ---


@app.post("/ingest/upload")
async def upload_video(
    file: UploadFile = File(...),
    context: str = Form("general")
):
    try:
        # 1. Governance / Validation
        task_metadata = VideoUploadTask(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=0,  # We'll update this after saving, or stream-count it
            context_tag=context
        )

        # 2. Write to Disk (IO Bound)
        file_location = f"{UPLOAD_DIR}/{task_metadata.task_id}.{task_metadata.filename.split('.')[-1]}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Update size for record
        task_metadata.size_bytes = os.path.getsize(file_location)

        # 3. Emit Domain Event (Kafka)
        event = ProcessingEvent(
            task_id=task_metadata.task_id,
            payload={
                "file_path": file_location,
                "context": context,
                "original_name": file.filename
            }
        )

        await producer.send_and_wait(
            TOPIC_TELEMETRY,
            value=event.json().encode('utf-8')
        )

        return {"task_id": task_metadata.task_id, "status": "accepted"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- QUERY: SSE STREAM ---


@app.get("/ingest/stream/{task_id}")
async def message_stream(request: Request, task_id: str):
    """
    Server-Sent Events Endpoint.
    Client keeps this open. We push data when Kafka receives it.
    """
    async def event_generator():
        q = asyncio.Queue()

        # Register connection
        if task_id not in active_connections:
            active_connections[task_id] = []
        active_connections[task_id].append(q)

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Wait for data from the Background Broadcaster
                # Timeout allows us to send keep-alive pings
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"

                    # If job is done, we can close the stream
                    if data.get("status") in ["completed", "failed"]:
                        yield "event: close\ndata: end\n\n"
                        break

                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        finally:
            # Cleanup
            active_connections[task_id].remove(q)
            if not active_connections[task_id]:
                del active_connections[task_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
