import json
from kafka import KafkaProducer

def get_producer(broker_url: str):
    return KafkaProducer(
        bootstrap_servers=broker_url,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )