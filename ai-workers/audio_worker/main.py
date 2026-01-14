import os
import json
import sys
import numpy as np
import parselmouth
import librosa
from kafka import KafkaConsumer, KafkaProducer
from vtoe_adapter import VToEAdapter

sys.path.append("/app")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "audio-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"

adapter = VToEAdapter()


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
    print(f"🎤 Audio Analyst (Parselmouth/Praat Mode) initializing...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-audio-worker-praat-v1",
        session_timeout_ms=60000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    vtoe = VToEAdapter()
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

            # 1. Load Audio with Praat
            try:
                # librosa.load extracts audio from video containers automatically
                y, sr = librosa.load(file_path, sr=None)
                sound = parselmouth.Sound(y, sampling_frequency=sr)
            except Exception as e:
                print(
                    f"   ...Librosa fallback failed, trying direct load: {e}")
                sound = parselmouth.Sound(file_path)

            total_duration = sound.get_total_duration()

            # 2. Calculate BASELINE PITCH (Global Average)
            # This is critical for Doc #110 (Stress Detection)
            full_pitch = sound.to_pitch()
            pitch_values = full_pitch.selected_array['frequency']
            # Remove 0s (unvoiced)
            voiced_pitches = pitch_values[pitch_values > 0]

            baseline_pitch = 0
            if len(voiced_pitches) > 0:
                baseline_pitch = np.mean(voiced_pitches)
                print(
                    f"   ...Calculated Baseline Pitch: {baseline_pitch:.2f} Hz")

            # 3. Analyze in Windows (e.g., 2 seconds)
            window_size = 2.0
            captions = []

            current_time = 0.0

            while current_time < total_duration:
                end_time = min(current_time + window_size, total_duration)

                # Extract part using Parselmouth
                # preserve_times=True keeps the original timestamps logic
                segment = sound.extract_part(
                    from_time=current_time, to_time=end_time)

                # --- Text Placeholder ---
                # Eventually, we will look up the text for this timestamp from Whisper
                current_text = None

                # Analyze
                signals = vtoe.analyze(
                    segment, text=current_text, baseline_pitch=baseline_pitch)

                if signals:
                    text_content = ", ".join(signals)
                    captions.append({
                        "start": current_time * 1000,
                        "end": end_time * 1000,
                        "text": f"VOCAL: {text_content}"
                    })

                current_time += window_size

            # Save VTT
            output_filename = f"{task_id}_audio.vtt"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            write_vtt(output_path, captions)

            print(f"✅ Audio VTT Saved: {output_path}")

            producer.send(DEST_TOPIC, {
                "task_id": task_id,
                "worker_type": "audio",
                "status": "completed",
                "download_url": f"/results/download/{task_id}/audio"
            })

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
