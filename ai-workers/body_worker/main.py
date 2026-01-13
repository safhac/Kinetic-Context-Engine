import os
import json
import cv2
import sys
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from kafka import KafkaConsumer, KafkaProducer
from body_worker.sensors import MediaPipeBodySensor

# Ensure imports work
sys.path.append(os.getcwd())

# --- CONFIGURATION ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "body-tasks")
DEST_TOPIC = os.getenv("DEST_TOPIC", "processed_signals")
RESULTS_DIR = "/app/media/results"  # Shared volume path

# MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws the skeleton on the frame.
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = rgb_image.copy()

    # Loop through the detected poses (usually just 1)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Convert the new API "NormalizedLandmark" object to the old Protobuf format
        # required by mp_drawing.draw_landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])

        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())

    return annotated_image


def main():
    print(f"💪 Body Worker (With Drawing) initializing...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-body-worker-drawing-v1",  # Changed group ID to force fresh start
        session_timeout_ms=60000,
        heartbeat_interval_ms=10000
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    sensor = MediaPipeBodySensor()
    print(f"✅ Body Worker Listening...")

    for message in consumer:
        try:
            payload = message.value
            task_id = payload.get("task_id")
            file_path = payload.get("file_path")

            if not file_path or not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            print(f"💪 Processing & Drawing: {task_id}")

            # 1. Setup Video Input
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # 2. Setup Video Output (The Result)
            output_filename = f"{task_id}_labeled.mp4"
            output_path = os.path.join(RESULTS_DIR, output_filename)

            # Try codec mp4v (widely supported in containers)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # --- SENSOR PROCESS ---
                # We perform detection directly here to get BOTH signals and drawing data.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Use the sensor's internal landmarker to detect
                detection_result = sensor.landmarker.detect(mp_image)

                # 1. Extract Signals (Re-using logic helper if available, or simplified here)
                signals = []
                # (Note: In a real refactor, you'd expose 'get_signals' from your sensor class)
                # For now, we will trust the sensor.process_frame logic if needed,
                # but to draw, we rely on detection_result.

                # Let's call the standard process to get the signals for the DB
                # This is slightly inefficient (double process) but safest for your current code structure
                # without breaking 'sensor.py'.
                signals = sensor.process_frame(frame, timestamp_ms)

                # 2. Draw Skeleton
                annotated_frame = frame.copy()
                if detection_result.pose_landmarks:
                    annotated_frame = draw_landmarks_on_image(
                        rgb_frame, detection_result)
                    annotated_frame = cv2.cvtColor(
                        annotated_frame, cv2.COLOR_RGB2BGR)

                # 3. Draw HUD (Text Overlay)
                if signals:
                    text = f"Detected: {', '.join(signals)}"
                    # Draw text with a black outline for visibility
                    cv2.putText(annotated_frame, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(annotated_frame, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Write frame to output video
                out.write(annotated_frame)

                # Send signals to Context Engine
                if signals:
                    producer.send(DEST_TOPIC, {
                        "task_id": task_id,
                        "timestamp": timestamp_ms,
                        "signals": signals,
                        "meta": {"worker": "body_worker"}
                    })

                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"   ...processed {frame_count} frames")

            cap.release()
            out.release()

            # --- FINAL STEP: Notify Context Engine about the Video ---
            artifact_msg = {
                "task_id": task_id,
                "timestamp": timestamp_ms,
                "signals": [],
                # URL for frontend
                "artifact_url": f"/media/results/{output_filename}",
                "status": "completed"
            }
            producer.send(DEST_TOPIC, artifact_msg)

            print(f"✅ Video Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
