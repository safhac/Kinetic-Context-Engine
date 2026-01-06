import os
import json
import logging
import sys
from kafka import KafkaConsumer, KafkaProducer

# ✅ CLEAN IMPORT: Relies on PYTHONPATH, not sys.path hacks
try:
    from deception_model import DeceptionModel
except ImportError:
    # Fallback for local testing if not running as a module
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
            group_id="kce-context-group-v2",  # Version 2 (Stateful)
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
    # We store a separate 'Brain' (DeceptionModel) for every user session.
    active_sessions = {}

    # 5. Main Loop
    logger.info("🧠 Brain is Active. Waiting for signals...")

    for message in consumer:
        try:
            payload = message.value

            # --- DATA ADAPTER ---
            # Handle single signal vs list of signals
            if "signals" in payload:
                incoming_signals = payload["signals"]
                session_id = payload.get("session_id", "unknown_session")
            else:
                incoming_signals = [payload]
                session_id = payload.get("metadata", {}).get(
                    "session_id", "default_session")

            # --- SESSION MANAGEMENT ---
            if session_id not in active_sessions:
                active_sessions[session_id] = DeceptionModel()
                logger.info(f"🆕 New Session Tracking: {session_id}")

            brain = active_sessions[session_id]

            # --- CORE LOGIC: UPDATE SCORE ---
            # We apply a small decay (cooling off) on every new message
            brain.decay()

            triggered_updates = []

            for sig_data in incoming_signals:
                sig_name = sig_data.get("signal")
                intensity = sig_data.get("intensity", 1.0)

                if sig_name:
                    # Update the math model
                    brain.update_score(sig_name, intensity)
                    triggered_updates.append(sig_name)

            # --- BROADCAST RESULT ---
            # We broadcast every update so the frontend can animate the stress bar
            output_payload = {
                "session_id": session_id,
                "timestamp": payload.get("timestamp"),
                "deception_score": round(brain.score, 2),
                "triggers": triggered_updates,
                "alert_level": "HIGH" if brain.score > 7.0 else "NORMAL"
            }

            producer.send(DEST_TOPIC, output_payload)

            # Console Feedback for Debugging
            if triggered_updates:
                log_icon = "🚨" if brain.score > 7.0 else "🧠"
                logger.info(
                    f"{log_icon} [Session: {session_id[:8]}] Score: {brain.score:.2f} | Detected: {triggered_updates}")

        except Exception as e:
            logger.error(f"⚠️ Error processing message: {e}")


if __name__ == "__main__":
    main()
