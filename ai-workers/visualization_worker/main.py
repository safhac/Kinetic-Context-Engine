import os
import json
import cv2
import sys
import redis
from kafka import KafkaConsumer, KafkaProducer

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = "render-tasks"  # New topic specifically for this
DEST_TOPIC = "interpreted_context"  # To notify the frontend when done
RESULTS_DIR = "/app/media/results"
UPLOAD_DIR = "/app/media/uploads"

# Redis for fetching the aggregated results (The "State")
redis_client = redis.Redis(host='redis', port=6379,
                           db=0, decode_responses=True)


def render_overlay(task_id, video_path, aggregated_data):
    """
    Burn JSON data onto the video frames.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Output file
    output_filename = f"{task_id}_final.mp4"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pre-process signals for fast lookup: { frame_number: ["Hands Raised", "Smirk"] }
    frame_map = {}
    for signal in aggregated_data:
        # Assuming signal has 'timestamp' (seconds) and 'label'
        ts = float(signal.get('timestamp', 0))
        label = signal.get('signal') or signal.get('type')
        start_frame = int(ts * fps)

        # Show the label for 1 second (30 frames) so it's readable
        for i in range(30):
            f_idx = start_frame + i
            if f_idx not in frame_map:
                frame_map[f_idx] = []
            frame_map[f_idx].append(label)

    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Look up what happened at this exact moment
        active_labels = list(
            set(frame_map.get(frame_count, [])))  # Deduplicate

        # 2. Draw HUD
        if active_labels:
            # Semi-transparent box
            overlay = frame.copy()
            box_height = 40 + (len(active_labels) * 40)
            cv2.rectangle(overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            y_pos = 50
            for label in active_labels:
                cv2.putText(frame, f"• {label}", (20, y_pos),
                            font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                y_pos += 40

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return f"/media/results/{output_filename}"


def main():
    print("🎨 Visualization Worker Started...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id="kce-viz-worker",
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    for message in consumer:
        task_id = message.value.get("task_id")
        file_path = message.value.get("file_path")

        print(f"🎨 Rendering final video for: {task_id}")

        # 1. Fetch all aggregated signals from Redis
        # (Assuming Context Engine stored them there)
        raw_signals = redis_client.get(f"results:{task_id}")
        if not raw_signals:
            print("⚠️ No signals found for this task.")
            signals = []
        else:
            signals = json.loads(raw_signals)

        # 2. Render
        try:
            download_url = render_overlay(task_id, file_path, signals)

            # 3. Notify Gateway
            producer.send(DEST_TOPIC, {
                "task_id": task_id,
                "status": "completed",
                "download_url": download_url
            })
            print(f"✅ Render Complete: {download_url}")

        except Exception as e:
            print(f"❌ Render Failed: {e}")


if __name__ == "__main__":
    main()
