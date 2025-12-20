import os
import json
import time
import logging
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# --- Configuration ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw-telemetry")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "kce-context-engine")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context-engine")

def start_consumer():
    logger.info(f"Connecting to {KAFKA_BROKER}...")
    
    # Retry Loop: Wait for Kafka to be ready
    while True:
        try:
            consumer = KafkaConsumer(
                SOURCE_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                # Deserializer converts bytes back to JSON automatically
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest' # Read from beginning if no history exists
            )
            logger.info("Connected to Kafka Bus.")
            break
        except NoBrokersAvailable:
            logger.warning("Waiting for Kafka... Retrying in 5s")
            time.sleep(5)

    # Main Event Loop
    for message in consumer:
        payload = message.value
        logger.info(f"Processing session: {payload.get('session_id')}")
        
        # Simulation of context processing (e.g., AI inference)
        time.sleep(0.1) 

if __name__ == "__main__":
    start_consumer()