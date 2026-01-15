import os
import json
import logging
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable  # <--- Added
from shared.schemas import GestureSignal, VideoProfile
from deception_model import DeceptionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kce-brain")

RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/media/results")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")


def main():
    logger.info("🧠 KCE-Brain (Unified Orchestrator + Context) starting...")

    # --- RETRY LOGIC ---
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                "raw_signals", "video_profiles",
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id="kce-brain-v1"
            )
            logger.info("✅ Brain connected to Kafka.")
        except NoBrokersAvailable:
            logger.warning("⏳ Kafka not ready. Retrying in 5s...")
            time.sleep(5)
    # -------------------

    profiles = {}
    signals = {}
    engine = DeceptionModel()

    for msg in consumer:
        data = msg.value
        task_id = data.get("task_id")
        if not task_id:
            continue

        if msg.topic == "video_profiles":
            profiles[task_id] = VideoProfile(**data)
            logger.info(f"📊 Profile stored for {task_id}")

        elif msg.topic == "raw_signals":
            sig = GestureSignal(**data)
            score = engine.analyze(sig)

            if task_id not in signals:
                signals[task_id] = []
            signals[task_id].append(sig)

            if task_id in profiles:
                write_final_vtt(task_id, profiles[task_id], signals[task_id])
                logger.info(
                    f"📝 Task {task_id} | Signal: {sig.worker_type} | Score: {score}%")


def write_final_vtt(task_id, profile, gesture_list):
    gesture_list.sort(key=lambda x: x.timestamp)
    vtt_lines = ["WEBVTT\n\n"]
    for sig in gesture_list:
        cue = sig.to_vtt_cue(profile)
        if cue:
            vtt_lines.append(cue)

    output_path = os.path.join(RESULTS_DIR, f"{task_id}.vtt")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("".join(vtt_lines))


if __name__ == "__main__":
    main()
