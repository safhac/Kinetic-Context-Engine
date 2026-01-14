import os
import json
import cv2
import sys
import mediapipe as mp
from kafka import KafkaConsumer, KafkaProducer
from body_worker.sensors import MediaPipeBodySensor
from body_worker.btoe_adapter import BToEAdapter

sys.path.append(os.getcwd())

# --- CONFIGURATION ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "body-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"


def ms_to_vtt_time(ms):
    """Converts milliseconds to VTT timestamp format HH:MM:SS.mmm"""
    seconds = int(ms / 1000)
    millis = int(ms % 1000)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)

    seconds = seconds % 60
    minutes = minutes % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def write_vtt(filename, captions):
    """Writes a list of captions to a .vtt file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for cap in captions:
            start = ms_to_vtt_time(cap['start'])
            end = ms_to_vtt_time(cap['end'])
            text = cap['text']
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def main():
    print(f"💪 Body Analyst (Subtitle Mode) initializing...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-body-worker-vtt-v2",
        session_timeout_ms=60000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = MediaPipeBodySensor()
    btoe = BToEAdapter()
    print(f"✅ Body Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            print(f"💪 Analyzing Body Language: {task_id}")

            cap = cv2.VideoCapture(file_path)

            # Tracking state for subtitles
            current_signal = None
            start_time = 0
            captions = []

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # 1. Run MediaPipe Detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = sensor.landmarker.detect(mp_image)

                # 2. Get Signals from BToE Adapter
                current_codes = []
                if detection_result.pose_landmarks:
                    # We usually only care about the first person detected [0]
                    raw_landmarks = detection_result.pose_landmarks[0]
                    current_codes = btoe.analyze_frame(raw_landmarks)

                # 3. Aggregation Logic for VTT
                # We prioritize the first detected code for the subtitle line
                active_code = current_codes[0] if current_codes else None

                if active_code != current_signal:
                    if current_signal:
                        # Close the previous caption
                        captions.append({
                            "start": start_time,
                            "end": timestamp_ms,
                            "text": f"BODY: {current_signal}"
                        })

                    if active_code:
                        # Start new caption
                        current_signal = active_code
                        start_time = timestamp_ms
                    else:
                        current_signal = None

                frame_count += 1
                if frame_count % 300 == 0:
                    print(f"   ...analyzed {frame_count} frames")

            # Close final caption if exists
            if current_signal:
                total_duration = cap.get(cv2.CAP_PROP_POS_MSEC)
                captions.append({
                    "start": start_time,
                    "end": total_duration,
                    "text": f"BODY: {current_signal}"
                })

            cap.release()

            # Save VTT
            output_filename = f"{task_id}_body.vtt"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            write_vtt(output_path, captions)

            print(f"✅ VTT Generated: {output_path}")

            # Notify Context Engine
            artifact_msg = {
                "task_id": task_id,
                "timestamp": timestamp_ms,
                "artifact_url": f"/media/results/{output_filename}",
                "artifact_type": "subtitle",
                "status": "completed"
            }
            producer.send(DEST_TOPIC, artifact_msg)

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
