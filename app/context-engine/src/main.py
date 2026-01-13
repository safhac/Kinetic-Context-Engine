import os
import json
import logging
import sys
from kafka import KafkaConsumer, KafkaProducer

# ✅ CLEAN IMPORT: Relies on PYTHONPATH, not sys.path hacks
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
# or 'interpreted_context' if using that
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "processed_signals")
DEST_TOPIC = os.getenv("DEST_TOPIC", "interpreted_context")


def main():
    logger.info("🚀 Context Engine (Deception Mode) Starting...")

    # 3. Initialize Kafka
    try:
        consumer = KafkaConsumer(
            SOURCE_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="kce-context-group-v3",  # Bumped version to force fresh read
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
            payload = message.value

            # --- DATA ADAPTER (FIXED) ---
            # 1. Extract Signals
            if "signals" in payload:
                incoming_signals = payload["signals"]
            else:
                # Handle single signal payload
                incoming_signals = [payload]

            # 2. Extract ID (Robust Check)
            # Workers might send 'task_id', 'session_id', or 'id'
            session_id = (
                payload.get("task_id") or
                payload.get("session_id") or
                payload.get("id") or
                payload.get("metadata", {}).get("session_id") or
                "unknown_session"
            )

            if session_id == "unknown_session":
                # Optional: print raw keys to debug if it fails again
                logger.warning(f"⚠️ payload keys: {list(payload.keys())}")

            # --- SESSION MANAGEMENT ---
            if session_id not in active_sessions:
                active_sessions[session_id] = DeceptionModel()
                logger.info(f"🆕 New Session Tracking: {session_id}")

            brain = active_sessions[session_id]

            # --- CORE LOGIC: UPDATE SCORE ---
            brain.decay()

            triggered_updates = []

            for sig_data in incoming_signals:
                # Handle both string (from list) and dict (from single) formats if necessary
                if isinstance(sig_data, str):
                    # If the list is just strings like ['lips_compressed', ...]
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

            # --- BROADCAST RESULT ---
            output_payload = {
                "session_id": session_id,  # Frontend looks for this!
                "timestamp": payload.get("timestamp"),
                "deception_score": round(brain.score, 2),
                "triggers": triggered_updates,
                "alert_level": "HIGH" if brain.score > 7.0 else "NORMAL"
            }

            producer.send(DEST_TOPIC, output_payload)

            if triggered_updates:
                log_icon = "🚨" if brain.score > 7.0 else "🧠"
                logger.info(
                    f"{log_icon} [Session: {session_id[:8]}] Score: {brain.score:.2f} | Detected: {len(triggered_updates)} signals")

        except Exception as e:
            logger.error(f"⚠️ Error processing message: {e}")


if __name__ == "__main__":
    main()
