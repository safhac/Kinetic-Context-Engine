import os
import json
import logging
import sys
from kafka import KafkaConsumer, KafkaProducer

try:
    from deception_model import DeceptionModel
except ImportError:
    from .deception_model import DeceptionModel

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kce-context-engine")

# 2. Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "processed_signals")
DEST_TOPIC = os.getenv("DEST_TOPIC", "interpreted_context")


def main():
    logger.info("🚀 Context Engine (Deception Mode) Starting...")

    # 3. Initialize Kafka
    try:
        consumer = KafkaConsumer(
            SOURCE_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="kce-context-group-v4",  # Bumped version
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"✅ Connected. Listening: {SOURCE_TOPIC}")
    except Exception as e:
        logger.critical(f"❌ Connection Failed: {e}")
        return

    # 4. Initialize Session Memory
    active_sessions = {}

    # 5. Main Loop
    logger.info("🧠 Brain is Active. Waiting for signals...")

    for message in consumer:
        try:
            payload = message.value  # <--- 'payload' IS DEFINED HERE

            # --- DATA ADAPTER ---
            if "signals" in payload:
                incoming_signals = payload["signals"]
            else:
                incoming_signals = [payload]

            # Robust ID Check
            task_id = (
                payload.get("task_id") or
                payload.get("session_id") or
                payload.get("id") or
                "unknown_session"
            )

            if task_id == "unknown_session":
                continue

            # --- SESSION MANAGEMENT ---
            if task_id not in active_sessions:
                active_sessions[task_id] = DeceptionModel()
                logger.info(f"🆕 New Session Tracking: {task_id}")

            brain = active_sessions[task_id]

            # --- CORE LOGIC ---
            brain.decay()
            triggered_updates = []

            for sig_data in incoming_signals:
                # Handle string vs dict formats
                if isinstance(sig_data, str):
                    sig_name = sig_data
                    intensity = 1.0
                elif isinstance(sig_data, dict):
                    sig_name = sig_data.get("signal") or sig_data.get("type")
                    intensity = sig_data.get("intensity", 1.0)
                else:
                    continue

                if sig_name:
                    brain.update_score(sig_name, intensity)
                    triggered_updates.append(sig_name)

            # --- PREPARE OUTPUT ---

            # Check if the worker sent us a finished video link
            download_link = payload.get(
                "download_url") or payload.get("artifact_url")

            worker_type = payload.get("worker_type", "general")

            output_payload = {
                "task_id": task_id,
                "worker_type": worker_type,
                "timestamp": payload.get("timestamp"),
                "deception_score": round(brain.score, 2),
                "triggers": triggered_updates,
                "alert_level": "HIGH" if brain.score > 7.0 else "NORMAL",
                "download_url": download_link
            }

            producer.send(DEST_TOPIC, output_payload)

            # Logging
            if download_link:
                logger.info(f"🔗 Forwarding Video Link for {task_id}")
            elif triggered_updates:
                log_icon = "🚨" if brain.score > 7.0 else "🧠"
                logger.info(
                    f"{log_icon} [{task_id[:8]}] Score: {brain.score:.2f} | Detected: {len(triggered_updates)}")

        except Exception as e:
            logger.error(f"⚠️ Error processing message: {e}")


if __name__ == "__main__":
    main()
