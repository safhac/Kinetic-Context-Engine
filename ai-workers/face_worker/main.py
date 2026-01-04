import os
import json
import base64
import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from sensors import MediaPipeFaceSensor, OpenFaceSensor

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw-telemetry")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed-signals")

def decode_frame(base64_string):
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def main():
    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Initialize Sensors via Interface
    sensors = [MediaPipeFaceSensor(), OpenFaceSensor()]
    print(f"👁️ Face Worker Started. Listening on {SOURCE_TOPIC}...")

    for message in consumer:
        payload = message.value
        session_id = payload.get("session_id")
        frame_data = payload.get("frame_data")
        timestamp = payload.get("timestamp", 0)

        if not frame_data:
            continue

        try:
            frame = decode_frame(frame_data)
            detected_signals = []

            # Polymorphic processing
            for sensor in sensors:
                signals = sensor.process_frame(frame, timestamp)
                detected_signals.extend(signals)

            # Produce signals if any found
            if detected_signals:
                output = {
                    "session_id": session_id,
                    "signals": detected_signals,
                    "meta": {"worker": "face_worker_v1"}
                }
                producer.send(DEST_TOPIC, output)
                print(f"✅ Sent {len(detected_signals)} signals for {session_id}")

        except Exception as e:
            print(f"❌ Error processing frame: {e}")

if __name__ == "__main__":
    main()