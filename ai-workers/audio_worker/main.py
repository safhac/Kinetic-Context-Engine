import os
import json
import sys
import time
from kafka import KafkaConsumer, KafkaProducer
# Assuming you have a basic AudioSensor class. If not, we can mock it.
from audio_worker.sensors import AudioSensor

sys.path.append(os.getcwd())

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "audio-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"

AUDIO_MEANING = {
    "shouting": "Aggressive / Urgent",
    "whisper": "Secretive / Uncertain",
    "laughter": "Positive / Mocking",
    "silence": "Pause",
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
    print(f"🎤 Audio Analyst (Subtitle Mode) initializing...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-audio-worker-vtt-v1",
        session_timeout_ms=60000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = AudioSensor()  # Ensure this class exists in audio_worker/sensors.py
    print(f"✅ Audio Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            print(f"🎤 Analyzing Audio: {task_id}")

            # Audio processing is usually faster than real-time
            # This function should return a list of segments: [{'start': 0, 'end': 1000, 'signal': 'neutral'}]
            # You might need to adjust this depending on your existing AudioSensor implementation
            analysis_segments = sensor.process_file(file_path)

            captions = []
            for seg in analysis_segments:
                signal = seg.get('signal', 'neutral')
                meaning = AUDIO_MEANING.get(signal, "Tone")
                captions.append({
                    "start": seg['start_ms'],
                    "end": seg['end_ms'],
                    "text": f"AUDIO: {signal.upper()} ({meaning})"
                })

            output_filename = f"{task_id}_audio.vtt"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            write_vtt(output_path, captions)

            print(f"✅ Audio VTT Saved: {output_path}")

            producer.send(DEST_TOPIC, {
                "task_id": task_id,
                "timestamp": 0,  # Audio summary doesn't have a single timestamp
                "artifact_url": f"/media/results/{output_filename}",
                "artifact_type": "subtitle",
                "status": "completed"
            })

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
