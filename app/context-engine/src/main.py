import os
import json
import logging
import sys
from kafka import KafkaConsumer, KafkaProducer

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kce-context-engine")

# 2. Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "processed_signals") # Input from Workers
DEST_TOPIC = os.getenv("DEST_TOPIC", "interpreted_context")   # Output to App

class RulesEngine:
    """
    The Semantic Brain.
    Maps atomic signals (e.g., 'eyebrow_raise') to Meaning (e.g., 'Friendliness').
    """
    def __init__(self, rules_path="src/rules.json"):
        try:
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
            logger.info(f"🧠 Knowledge Base Loaded: {len(self.rules)} rules active.")
        except Exception as e:
            logger.error(f"❌ Failed to load rules: {e}")
            self.rules = []

    def evaluate(self, signals):
        interpretations = []
        # Convert list of signals to a quick-lookup dictionary: {'signal_name': intensity}
        detected_map = {s['signal']: s.get('intensity', 1.0) for s in signals}

        for rule in self.rules:
            # Logic: ALL triggers in the rule must be present in the detected signals
            # Example: Rule "Turtling" requires "shoulders_up" AND "head_down"
            triggers = rule.get('triggers', [])
            if not triggers: continue

            if all(t in detected_map for t in triggers):
                # Calculate average intensity
                avg_intensity = sum(detected_map[t] for t in triggers) / len(triggers)
                
                # Check threshold
                if avg_intensity >= rule.get('threshold', 0.0):
                    interpretations.append({
                        "meaning": rule['meaning'],
                        "code": rule.get('rule_code', 'unknown'),
                        "confidence": avg_intensity,
                        "triggers": triggers
                    })
        return interpretations

def main():
    logger.info("🚀 Context Engine Starting...")
    
    # 3. Initialize Kafka Consumer (The Listener)
    # We use 'group_id' to ensure we are a persistent consumer.
    try:
        consumer = KafkaConsumer(
            SOURCE_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="kce-context-group-v1",
            auto_offset_reset='earliest', # Important: Read from start if we missed data
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        logger.info(f"✅ Consumer connected to {SOURCE_TOPIC}")
    except Exception as e:
        logger.critical(f"❌ Failed to connect Consumer: {e}")
        return

    # 4. Initialize Kafka Producer (The Speaker)
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"✅ Producer connected. Broadcasting to {DEST_TOPIC}")
    except Exception as e:
        logger.critical(f"❌ Failed to connect Producer: {e}")
        return

    # 5. Initialize Brain
    brain = RulesEngine()

    # 6. Main Processing Loop
    logger.info("🧠 Waiting for signals...")
    
    for message in consumer:
        try:
            payload = message.value
            
            # Extract basic info
            session_id = payload.get("session_id")
            signals = payload.get("signals", [])
            
            if not signals:
                continue

            # --- STEP A: PERCEIVE ---
            meanings = brain.evaluate(signals)

            if meanings:
                # --- STEP B: INTERPRET ---
                output_payload = {
                    "session_id": session_id,
                    "timestamp": payload.get("timestamp"),
                    "context": meanings,
                    "meta": {
                        "engine_version": "2.0",
                        "source_signals": len(signals)
                    }
                }

                # --- STEP C: BROADCAST ---
                producer.send(DEST_TOPIC, output_payload)
                
                # Log the "Thought"
                for m in meanings:
                    logger.info(f"💡 [Session: {session_id[:8]}] Detected: {m['meaning']} (Confidence: {m['confidence']:.2f})")

        except Exception as e:
            logger.error(f"⚠️ Error processing message: {e}")

if __name__ == "__main__":
    main()