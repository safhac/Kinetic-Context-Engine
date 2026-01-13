import os
import json
import cv2
import time
from kafka import KafkaConsumer, KafkaProducer
from sensors import MediaPipeFaceSensor

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "face-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "interpreted_context")


def main():
    # 1. FIX: Added group_id to ensure robust consumption
    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id='kce-face-worker',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Initialize Sensor
    # Note: Ensure you are using the fixed sensors.py with static_image_mode=True
    sensor = MediaPipeFaceSensor()

    print(f"👁️ Face Worker Started. Listening on {SOURCE_TOPIC}...")

    for message in consumer:
        payload = message.value
        task_id = payload.get("task_id") or payload.get("session_id")
        # 2. FIX: specific key for file processing
        file_path = payload.get("file_path")

        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ Invalid or missing file path: {file_path}")
            continue

        print(f"👁️ Processing Face Task: {session_id}")

        try:
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0
            all_signals = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps
                signals = sensor.process_frame(frame, timestamp)
                all_signals.extend(signals)

                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"   ...processed {frame_count} frames")

            cap.release()

            if all_signals:
                output = {
                    "task_id": task_id,  # <--- FIX: Send back task_id
                    "signals": all_signals,
                    "type": "face_analysis",
                    "meta": {"worker": "face_worker_v1", "frames_processed": frame_count}
                }
                producer.send(DEST_TOPIC, output)
                print(
                    f"✅ Finished Face Task: {task_id} ({len(all_signals)} signals found)")
            else:
                print(f"⚠️ No face signals detected for {task_id}")

        except Exception as e:
            print(f"❌ Error processing video: {e}")


if __name__ == "__main__":
    main()
