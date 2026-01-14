import os
import json
import cv2
import sys
from kafka import KafkaConsumer, KafkaProducer
from face_worker.sensors import MediaPipeFaceSensor

sys.path.append(os.getcwd())

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "face-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"

# --- MEANING MAP ---
FACE_MEANING = {
    "smile": "Positive / Agreement",
    "frown": "Negative / Disagreement",
    "surprise": "Shock / Unexpected",
    "neutral": "Baseline"
}


def ms_to_vtt_time(ms):
    seconds = int(ms / 1000)
    millis = int(ms % 1000)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    return f"{hours:02}:{minutes % 60:02}:{seconds % 60:02}.{millis:03}"


def write_vtt(filename, captions):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for cap in captions:
            f.write(
                f"{ms_to_vtt_time(cap['start'])} --> {ms_to_vtt_time(cap['end'])}\n")
            f.write(f"{cap['text']}\n\n")


def main():
    print(f"🙂 Face Analyst (Subtitle Mode) initializing...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-face-worker-vtt-v1",
        session_timeout_ms=60000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = MediaPipeFaceSensor()
    print(f"✅ Face Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            print(f"🙂 Analyzing Face: {task_id}")

            cap = cv2.VideoCapture(file_path)

            current_signal = None
            start_time = 0
            captions = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Process Frame
                signals_list = sensor.process_frame(frame, timestamp_ms)

                # Robust extraction
                active_gesture = None
                if signals_list:
                    raw = signals_list[0]
                    active_gesture = raw.get('signal') if isinstance(
                        raw, dict) else str(raw)

                # Aggregation Logic
                if active_gesture != current_signal:
                    if current_signal:
                        meaning = FACE_MEANING.get(
                            current_signal, "Expression")
                        captions.append({
                            "start": start_time,
                            "end": timestamp_ms,
                            "text": f"FACE: {current_signal.upper()} ({meaning})"
                        })

                    if active_gesture:
                        current_signal = active_gesture
                        start_time = timestamp_ms
                    else:
                        current_signal = None

                frame_count += 1
                if frame_count % 300 == 0:
                    print(f"   ...analyzed {frame_count} frames")

            # Close final caption
            if current_signal:
                captions.append({
                    "start": start_time,
                    "end": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "text": f"FACE: {current_signal.upper()}"
                })

            cap.release()

            output_filename = f"{task_id}_face.vtt"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            write_vtt(output_path, captions)
            print(f"✅ Face VTT Saved: {output_path}")

            producer.send(DEST_TOPIC, {
                "task_id": task_id,
                "timestamp": timestamp_ms,
                "artifact_url": f"/media/results/{output_filename}",
                "artifact_type": "subtitle",
                "status": "completed"
            })

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
