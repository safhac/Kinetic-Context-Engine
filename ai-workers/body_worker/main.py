from body_worker.sensors import MediaPipeBodySensor
import os
import json
import cv2
import time
import sys
from kafka import KafkaConsumer, KafkaProducer

# Ensure we can import from the sibling directory
sys.path.append(os.getcwd())

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
# UPDATE: Default to 'body-tasks' to match your Ingestion Service
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "body-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")


def main():
    print(f"💪 Body Worker initializing on {KAFKA_BROKER}...")

    # Consumer
    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-body-worker",
        # CRITICAL: Prevent "Zombie Loops" on long videos
        session_timeout_ms=60000,    # 60s silence allowed
        heartbeat_interval_ms=10000,  # Heartbeat every 10s
        max_poll_interval_ms=900000  # 15 mins max processing time
    )

    # Producer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = MediaPipeBodySensor()
    print(f"✅ Body Worker Listening on {SOURCE_TOPIC}...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            # Validation
            if not file_path:
                print("⚠️ Received task without file_path. Skipping.")
                continue

            if not os.path.exists(file_path):
                print(f"⚠️ File not found via Shared Volume: {file_path}")
                continue

            print(f"💪 Processing Body Task: {task_id}")

            # Open Video File
            cap = cv2.VideoCapture(file_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get timestamp in milliseconds
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Extract Signals
                signals = sensor.process_frame(frame, timestamp)

                if signals:
                    output = {
                        "task_id": task_id,
                        "timestamp": timestamp,
                        "signals": signals,
                        "meta": {"worker": "body_worker_v1"}
                    }
                    producer.send(DEST_TOPIC, output)

                frame_count += 1
                # Optional: Logging progress every 60 frames to keep logs clean
                if frame_count % 60 == 0:
                    print(f"   ...processed {frame_count} frames")

            cap.release()
            producer.flush()
            print(f"✅ Finished Body Task: {task_id}")

        except Exception as e:
            print(f"❌ Error processing task: {e}")


if __name__ == "__main__":
    main()
