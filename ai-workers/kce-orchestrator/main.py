import json
import os
from kafka import KafkaConsumer
from shared.schemas import GestureSignal, VideoProfile

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
RESULTS_DIR = "/app/media/results"


def main():
    consumer = KafkaConsumer(
        "raw_signals", "video_profiles",
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    # Memory store: { task_id: { "profile": VideoProfile, "signals": [GestureSignal] } }
    active_tasks = {}

    print("🚀 Merger Service Active. Awaiting signals...")

    for msg in consumer:
        data = msg.value
        task_id = data.get("task_id")
        if not task_id:
            continue

        if task_id not in active_tasks:
            active_tasks[task_id] = {"profile": None, "signals": []}

        # 1. Capture the Video Profile (Dimensions/Audio status)
        if msg.topic == "video_profiles":
            active_tasks[task_id]["profile"] = VideoProfile(**data)
            print(f"📊 Profile received for {task_id}")

        # 2. Capture Raw Gesture Signals
        elif msg.topic == "raw_signals":
            signal = GestureSignal(**data)
            active_tasks[task_id]["signals"].append(signal)

        # 3. Logic: Update VTT immediately for "Real-time" feel
        profile = active_tasks[task_id]["profile"]
        if profile:
            render_trinity_vtt(
                task_id, profile, active_tasks[task_id]["signals"])


def render_trinity_vtt(task_id, profile, signals):
    """Uses the schemas.py logic to build the styled VTT."""
    # Sort by time so VTT blocks are in order
    signals.sort(key=lambda x: x.timestamp)

    vtt_content = ["WEBVTT\n\n"]
    for sig in signals:
        # This calls the logic we put in shared/schemas.py
        cue = sig.to_vtt_cue(profile)
        if cue:
            vtt_content.append(cue)

    # Save as {task_id}.vtt for auto-matching in video players
    file_path = os.path.join(RESULTS_DIR, f"{task_id}.vtt")
    with open(file_path, "w") as f:
        f.write("".join(vtt_content))
