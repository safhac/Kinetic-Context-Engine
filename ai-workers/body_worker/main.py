import os
import json
import cv2
import sys
import mediapipe as mp
from kafka import KafkaConsumer, KafkaProducer
from body_worker.sensors import MediaPipeBodySensor

sys.path.append(os.getcwd())

# --- CONFIGURATION ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "body-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"

# --- SIMPLE DICTIONARY (Mockup for now) ---
# In a real app, this might come from a DB or config file
GESTURE_MEANING = {
    "hands_on_face": "Stress / Hiding emotion",
    "arms_crossed": "Defensive / Closed off",
    "fidgeting": "Nervousness / Deception",
    "lean_forward": "Interest / Aggression",
    "neutral": "Baseline"
}


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
        group_id="kce-body-worker-vtt-v1",
        session_timeout_ms=60000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = MediaPipeBodySensor()
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
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Helper to group continuous signals
            current_signal = None
            start_time = 0
            captions = []

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Process Frame (No drawing, just data)
                signals_list = sensor.process_frame(frame, timestamp_ms)

                # Extract primary signal (simplify to the first one for subtitles)
                # In the future: handle multiple concurrent signals
                active_gesture = None
                if signals_list:
                    # Robust extraction (fixing your previous error)
                    raw_signal = signals_list[0]
                    active_gesture = raw_signal.get('signal') if isinstance(
                        raw_signal, dict) else str(raw_signal)

                # --- AGGREGATION LOGIC ---
                # Only write a subtitle if the gesture CHANGES or ENDS
                if active_gesture != current_signal:
                    if current_signal:
                        # Close the previous caption
                        meaning = GESTURE_MEANING.get(
                            current_signal, "Unclassified gesture")
                        captions.append({
                            "start": start_time,
                            "end": timestamp_ms,
                            "text": f"BODY: {current_signal.upper()} ({meaning})"
                        })

                    # Start new caption
                    if active_gesture:
                        current_signal = active_gesture
                        start_time = timestamp_ms
                    else:
                        current_signal = None

                frame_count += 1
                if frame_count % 300 == 0:
                    print(f"   ...analyzed {frame_count} frames")

            # Close final caption if exists
            if current_signal:
                total_duration = cap.get(cv2.CAP_PROP_POS_MSEC)
                meaning = GESTURE_MEANING.get(
                    current_signal, "Unclassified gesture")
                captions.append({
                    "start": start_time,
                    "end": total_duration,
                    "text": f"BODY: {current_signal.upper()} ({meaning})"
                })

            cap.release()

            # Save VTT
            output_filename = f"{task_id}_body.vtt"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            write_vtt(output_path, captions)

            print(f"✅ VTT Generated: {output_path}")

            # Notify Context Engine (Send the VTT path, NOT a video path)
            artifact_msg = {
                "task_id": task_id,
                "timestamp": timestamp_ms,
                "signals": [],
                "artifact_url": f"/media/results/{output_filename}",
                "artifact_type": "subtitle",
                "status": "completed"
            }
            producer.send(DEST_TOPIC, artifact_msg)

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
