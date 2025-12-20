import os
import json
import time
import logging
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
SOURCE_TOPIC = os.getenv("SOURCE_TOPIC", "raw-telemetry")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "kce-context-engine")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context-engine")

def start_consumer():
    logger.info(f"Connecting to {KAFKA_BROKER}...")
    while True:
        try:
            consumer = KafkaConsumer(
                SOURCE_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest'
            )
            logger.info("Connected.")
            break
        except NoBrokersAvailable:
            logger.warning("Waiting for Kafka...")
            time.sleep(5)

    for message in consumer:
        payload = message.value
        logger.info(f"Processing session: {payload.get('session_id')}")
        # Simulation of context processing
        time.sleep(0.1) 

if __name__ == "__main__":
    start_consumer()