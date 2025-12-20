from fastapi import FastAPI
from shared.schemas import ImageTask
import json
import os
# Placeholder for Kafka Producer (e.g., confluent-kafka or kafka-python)
# from kafka import KafkaProducer 

app = FastAPI()

# KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
# producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)

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