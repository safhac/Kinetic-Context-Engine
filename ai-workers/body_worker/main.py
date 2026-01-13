from body_worker.sensors import MediaPipeBodySensor
import os
import json
import base64
import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import sys

# Ensure we can import from the sibling directory
sys.path.append(os.getcwd())
# Import the BODY sensor we just created

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
# Listening to the same raw stream
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw_telemetry")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")


def decode_frame(base64_string):
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def main():
    print(f"💪 Body Worker initializing on {KAFKA_BROKER}...")

    # Consumer
    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        # CRITICAL: Use a DIFFERENT group_id than the Face Worker.
        # This ensures Kafka sends the message to BOTH workers (Fan-Out), not just one.
        group_id="kce-body-worker"
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
            frame_data = payload.get("frame_data")
            session_id = payload.get("session_id")
            timestamp = payload.get("timestamp", 0)

            if not frame_data:
                continue

            frame = decode_frame(frame_data)

            # Extract Signals
            signals = sensor.process_frame(frame, timestamp)

            if signals:
                output = {
                    "session_id": session_id,
                    "signals": signals,
                    "meta": {"worker": "body_worker_v1"}
                }
                producer.send(DEST_TOPIC, output)
                print(f"📤 Sent {len(signals)} BODY signals for {session_id}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
