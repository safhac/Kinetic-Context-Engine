from shared.schemas import GestureSignal
import os
import json
import cv2
import sys
import mediapipe as mp
from kafka import KafkaConsumer, KafkaProducer

# 1. Point to shared schemas
sys.path.append("/app")

# --- CONFIGURATION ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "body-tasks")
RAW_SIGNAL_TOPIC = "raw_signals"
STATUS_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")


def main():
    print(f"💪 Body Analyst initializing (Signal Mode)...")

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-body-worker-v3"
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    from body_worker.sensors import MediaPipeBodySensor
    from body_worker.btoe_adapter import BToEAdapter

    sensor = MediaPipeBodySensor()
    btoe = BToEAdapter()

    print(f"✅ Body Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                continue

            print(f"💪 Analyzing Body: {task_id}")
            cap = cv2.VideoCapture(file_path)

            # 2. Tracking state for signal change (Debouncing)
            last_signal = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Run Detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = sensor.landmarker.detect(mp_image)

                if detection_result.pose_landmarks:
                    raw_landmarks = detection_result.pose_landmarks[0]
                    current_codes = btoe.analyze_frame(raw_landmarks)

                    if current_codes:
                        active_code = current_codes[0]

                        # 3. Only emit if the gesture has changed (Avoid flooding Kafka)
                        if active_code != last_signal:
                            signal = GestureSignal(
                                task_id=task_id,
                                worker_type="body",
                                timestamp=timestamp_sec,
                                text=active_code,
                                confidence=1.0
                            )
                            producer.send(RAW_SIGNAL_TOPIC,
                                          signal.model_dump())
                            last_signal = active_code
                    else:
                        last_signal = None

            cap.release()

            # 4. Notify completion
            producer.send(STATUS_TOPIC, {
                "task_id": task_id,
                "worker_type": "body",
                "status": "signals_sent"
            })

        except Exception as e:
            print(f"❌ Error in Body Worker: {e}")


if __name__ == "__main__":
    main()
