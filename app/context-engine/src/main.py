import os
import json
import time
import base64
import logging
import cv2
import numpy as np
import mediapipe as mp
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import NoBrokersAvailable

# --- Configuration ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw-telemetry")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "kce-context-engine")
LAG_THRESHOLD_SEC = 0.1  # 100ms threshold for skip-logic

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("context-engine")

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Initialize models once (Heavy operation)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

def decode_frame(base64_string):
    """Converts Base64 string back to OpenCV image."""
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Frame decoding failed: {e}")
        return None

def process_frame(frame, session_id):
    """
    Orchestrates MediaPipe Inference.
    """
    # MediaPipe requires RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Face Mesh Analysis
    face_results = face_mesh.process(frame_rgb)
    face_count = 0
    if face_results.multi_face_landmarks:
        face_count = len(face_results.multi_face_landmarks)

    # 2. Pose Analysis
    pose_results = pose.process(frame_rgb)
    pose_detected = "Yes" if pose_results.pose_landmarks else "No"

    logger.info(f"Session: {session_id} | Face: {face_count} | Pose: {pose_detected}")

def start_consumer():
    logger.info(f"Connecting to {KAFKA_BROKER}...")
    
    # Resilience: Retry Loop
    consumer = None
    while not consumer:
        try:
            consumer = KafkaConsumer(
                SOURCE_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',  # Real-time systems prefer latest
                enable_auto_commit=False     # CRITICAL: We commit manually
            )
            logger.info("Connected to Kafka Bus.")
        except NoBrokersAvailable:
            logger.warning("Waiting for Kafka... Retrying in 5s")
            time.sleep(5)

    # --- Main Event Loop ---
    try:
        for message in consumer:
            payload = message.value
            metadata = payload.get("metadata", {})
            session_id = payload.get("session_id", "unknown")
            timestamp = metadata.get("timestamp", time.time())
            
            # --- 1. Latency Governance (Skip-Frame Logic) ---
            current_time = time.time()
            lag = current_time - timestamp
            
            if lag > LAG_THRESHOLD_SEC:
                logger.warning(f"High Lag Detected ({lag:.3f}s). Skipping frame to recover.")
                # We still commit this offset to mark it as "handled" (skipped)
                consumer.commit() 
                continue

            # --- 2. Decode & Process ---
            frame_data = payload.get("frame_data")
            if frame_data:
                img = decode_frame(frame_data)
                if img is not None:
                    process_frame(img, session_id)

            # --- 3. Atomic Commit ---
            # Commit processing of this specific batch to ensure At-Least-Once delivery
            consumer.commit() 
            
    except Exception as e:
        logger.critical(f"Consumer crashed: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    start_consumer()