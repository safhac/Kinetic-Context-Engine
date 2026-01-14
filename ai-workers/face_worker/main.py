from shared.schemas import GestureSignal
import os
import json
import cv2
import sys
import mediapipe as mp
from kafka import KafkaConsumer, KafkaProducer

# 1. Point to shared schemas
sys.path.append("/app")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "face-tasks")
RAW_SIGNAL_TOPIC = "raw_signals"
STATUS_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")


def main():
    print(f"🙂 Face Analyst initializing (Grace Mode)...")

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-face-worker-v3"
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    from face_worker.sensors import MediaPipeFaceSensor
    from face_worker.ftoe_adapter import FToEAdapter

    sensor = MediaPipeFaceSensor()
    ftoe = FToEAdapter()

    print(f"✅ Face Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                continue

            cap = cv2.VideoCapture(file_path)

            # --- GRACE LOGIC STATE ---
            last_signal = None
            last_seen_time = 0
            GRACE_PERIOD = 0.5

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = sensor.landmarker.detect(mp_image)

                active_code = None
                if detection_result.face_landmarks:
                    raw_landmarks = detection_result.face_landmarks[0]
                    current_codes = ftoe.analyze_frame(
                        raw_landmarks, frame=frame)
                    if current_codes:
                        active_code = current_codes[0]

                # --- HYSTERESIS ENGINE ---
                if active_code:
                    if active_code != last_signal:
                        signal = GestureSignal(
                            task_id=task_id,
                            worker_type="face",
                            timestamp=timestamp_sec,
                            text=active_code,
                            confidence=1.0
                        )
                        producer.send(RAW_SIGNAL_TOPIC, signal.model_dump())
                        last_signal = active_code

                    last_seen_time = timestamp_sec
                else:
                    if last_signal and (timestamp_sec - last_seen_time > GRACE_PERIOD):
                        last_signal = None

            cap.release()
            producer.send(STATUS_TOPIC, {
                          "task_id": task_id, "worker_type": "face", "status": "signals_sent"})

        except Exception as e:
            print(f"❌ Error in Face Worker: {e}")


if __name__ == "__main__":
    main()
