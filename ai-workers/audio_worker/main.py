from shared.schemas import GestureSignal
import os
import json
import sys
import numpy as np
import parselmouth
import librosa
from kafka import KafkaConsumer, KafkaProducer

# 1. Point to the shared schemas
sys.path.append("/app")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "audio-tasks")
RAW_SIGNAL_TOPIC = "raw_signals"  # The new centralized stream
STATUS_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")

# We removed VToEAdapter and VTT writing functions as they are now
# handled by the Orchestrator/Schemas logic.


def main():
    print(f"🎤 Audio Analyst initializing (Signal Mode)...")

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-audio-worker-v2"
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Note: Import your existing analysis logic/adapter here
    from vtoe_adapter import VToEAdapter
    vtoe = VToEAdapter()

    print(f"✅ Audio Analyst Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                continue

            print(f"🎤 Analyzing Audio: {task_id}")

            y, sr = librosa.load(file_path, sr=None)
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            total_duration = sound.get_total_duration()

            # Baseline Pitch Calculation
            full_pitch = sound.to_pitch()
            voiced_pitches = full_pitch.selected_array['frequency'][full_pitch.selected_array['frequency'] > 0]
            baseline_pitch = np.mean(voiced_pitches) if len(
                voiced_pitches) > 0 else 0

            window_size = 2.0
            current_time = 0.0

            while current_time < total_duration:
                end_time = min(current_time + window_size, total_duration)
                segment = sound.extract_part(
                    from_time=current_time, to_time=end_time)

                # Analyze signals
                signals = vtoe.analyze(
                    segment, text=None, baseline_pitch=baseline_pitch)

                if signals:
                    # 2. UTILIZE GESTURE CLASS
                    # We create a signal for each detection in this window
                    signal = GestureSignal(
                        task_id=task_id,
                        worker_type="audio",
                        timestamp=current_time,
                        text=", ".join(signals),
                        confidence=1.0  # Or map from your analysis
                    )

                    # 3. EMIT TO RAW TOPIC
                    producer.send(RAW_SIGNAL_TOPIC, signal.model_dump())

                current_time += window_size

            # 4. Notify completion (Status only, no download URL here)
            producer.send(STATUS_TOPIC, {
                "task_id": task_id,
                "worker_type": "audio",
                "status": "signals_sent"
            })

        except Exception as e:
            print(f"❌ Error in Audio Worker: {e}")


if __name__ == "__main__":
    main()
