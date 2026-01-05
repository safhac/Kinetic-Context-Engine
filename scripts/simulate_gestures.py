import json
import time
import random
import uuid
from kafka import KafkaProducer

# Configuration
KAFKA_BROKER = "localhost:9092" # Assumes running from host, mapped to 9092
DEST_TOPIC = "processed_signals"

# The "Vocabulary" of atomic signals our Workers *will* eventually produce
POSSIBLE_SIGNALS = [
    "eyebrow_raise", "lip_compression", "chin_thrust", 
    "head_tilt", "arms_crossed", "hand_raise", 
    "shoulders_up", "head_down", "hand_on_face"
]

def main():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"❌ Failed to connect to Kafka at {KAFKA_BROKER}. Is it running? Error: {e}")
        return

    session_id = str(uuid.uuid4())
    print(f"🎭 Starting Simulation. Session ID: {session_id}")
    print(f"📡 Injecting signals into topic: '{DEST_TOPIC}'...\n")

    try:
        while True:
            # 1. Randomly pick 1-3 signals to simulate a complex moment
            # e.g., "shoulders_up" + "head_down" = Turtling
            current_signals = random.sample(POSSIBLE_SIGNALS, k=random.randint(1, 3))
            
            # 2. Construct the payload usually sent by Workers
            timestamp = time.time()
            payload = {
                "session_id": session_id,
                "timestamp": timestamp,
                "signals": [
                    {"signal": name, "intensity": round(random.uniform(0.6, 1.0), 2), "source": "sim_script"}
                    for name in current_signals
                ],
                "meta": {"worker": "simulation_script"}
            }

            # 3. Send
            producer.send(DEST_TOPIC, payload)
            
            # 4. Log
            print(f"⚡ Sent: {current_signals}")
            time.sleep(2) # Wait 2 seconds between gestures

    except KeyboardInterrupt:
        print("\n🛑 Simulation Stopped.")

if __name__ == "__main__":
    main()