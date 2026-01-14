import os
import json
import logging
from kafka import KafkaConsumer
from schemas import GestureSignal, VideoProfile  # Duplicated or shared

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kce-orchestrator")

# --- CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
RESULTS_DIR = "/app/media/results"


def main():
    logger.info("🧠 Orchestrator starting...")

    consumer = KafkaConsumer(
        "raw_signals", "video_profiles",
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-orchestrator-v1"
    )

    # State stores
    profiles = {}  # task_id -> VideoProfile
    signals = {}  # task_id -> List[GestureSignal]

    for msg in consumer:
        data = msg.value
        task_id = data.get("task_id")
        if not task_id:
            continue

        # 1. Store Video Profile
        if msg.topic == "video_profiles":
            profiles[task_id] = VideoProfile(**data)
            logger.info(f"📊 Profile stored for {task_id}")

        # 2. Store Gesture Signals
        elif msg.topic == "raw_signals":
            if task_id not in signals:
                signals[task_id] = []
            signals[task_id].append(GestureSignal(**data))

            # 3. Trigger VTT Write (Real-time update)
            if task_id in profiles:
                write_final_vtt(task_id, profiles[task_id], signals[task_id])


def write_final_vtt(task_id, profile, gesture_list):
    """Assembles the Trinity VTT based on the Contract."""
    # Ensure chronological order
    gesture_list.sort(key=lambda x: x.timestamp)

    vtt_lines = ["WEBVTT\n\n"]

    for sig in gesture_list:
        # utilize the logic we created in schemas.py
        # This handles: 1. Dismissal 2. 2s Persistence 3. Colors 4. Top-Stacking
        cue = sig.to_vtt_cue(profile)
        if cue:
            vtt_lines.append(cue)

    # Save as {task_id}.vtt for auto-matching in players
    output_path = os.path.join(RESULTS_DIR, f"{task_id}.vtt")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("".join(vtt_lines))

    logger.info(f"📝 VTT Updated: {output_path}")


if __name__ == "__main__":
    main()
