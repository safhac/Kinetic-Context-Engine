import os
import json
import time
import base64
import logging
import cv2
import numpy as np
import mediapipe as mp
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# --- Configuration ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw_telemetry")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "kce-context-group")
LAG_THRESHOLD_SEC = 0.5  # Skip frames older than 0.5s to prevent backlog

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("context-engine")

# --- Robust MediaPipe Initialization ---
# We wrap this in a try-except block so the container doesn't crash 
# if it fails to get a GPU context (common EGL error in Docker).
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

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
    logger.info("✅ MediaPipe Models Loaded Successfully")
except Exception as e:
    logger.error(f"❌ MediaPipe Failed to Load (Running in limited mode): {e}")
    face_mesh = None
    pose = None

def decode_frame(base64_string):
    """Converts Base64 string back to OpenCV image."""
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Frame decoding failed: {e}")
        return None

def process_frame(frame, session_id, timestamp):
    """
    Orchestrates MediaPipe Inference.
    """
    if not face_mesh or not pose:
        logger.warning("Skipping processing: AI models not loaded.")
        return

    try:
        # MediaPipe requires RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Face Mesh Analysis (The Face Recognition Logic)
        face_results = face_mesh.process(frame_rgb)
        face_count = 0
        if face_results.multi_face_landmarks:
            face_count = len(face_results.multi_face_landmarks)

        # 2. Pose Analysis
        pose_results = pose.process(frame_rgb)
        pose_detected = "Yes" if pose_results.pose_landmarks else "No"

        # Calculate processing lag for observability
        process_lag = time.time() - timestamp
        
        logger.info(f"Session: {session_id} | Face: {face_count} | Pose: {pose_detected} | Lag: {process_lag:.3f}s")

    except Exception as e:
        logger.error(f"Inference Error: {e}")

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
                auto_offset_reset='latest',  # Start at the end of the queue
                enable_auto_commit=False     # CRITICAL: Manual commits enabled
            )
            logger.info("✅ Connected to Kafka Bus.")
        except NoBrokersAvailable:
            logger.warning("⏳ Waiting for Kafka... Retrying in 2s")
            time.sleep(2)

    # --- Main Event Loop ---
    print("🎧 Consumer Ready. Waiting for frames...", flush=True)
    
    for message in consumer:
        
        
        try:
            payload = message.value
            
            # 1. Extract Metadata (RESTORED)
            metadata = payload.get("metadata", {})
            session_id = payload.get("session_id", "unknown")
            timestamp = metadata.get("timestamp", time.time())
            
            # 2. Latency Governance (Skip-Frame Logic) (RESTORED)
            # current_lag = time.time() - timestamp
            
            # if current_lag > LAG_THRESHOLD_SEC:
            #     logger.warning(f"⏩ High Lag ({current_lag:.3f}s). Skipping frame to recover.")
            #     # We commit this offset to mark it as "handled" (skipped) so we don't read it again
            #     consumer.commit() 
            #     continue

            # 3. Decode & Process
            frame_data = payload.get("frame_data")
            if frame_data:
                img = decode_frame(frame_data)
                if img is not None:
                    process_frame(img, session_id, timestamp)

            # 4. Atomic Commit (RESTORED)
            # Only commit after successful processing to ensure At-Least-Once delivery
            consumer.commit() 
            
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
            # Optional: Decide if you want to commit on error or retry
            # consumer.commit() 

if __name__ == "__main__":
    start_consumer()