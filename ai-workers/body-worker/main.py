import time
import os
import sys

# Add root to path to import shared
sys.path.append(os.getcwd())
from shared.schemas import TaskResult

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown-worker")

def process_task():
    print(f"[{SERVICE_NAME}] connect to Kafka...")
    while True:
        # Simulate processing
        print(f"[{SERVICE_NAME}] Processing task...")
        time.sleep(5) 
        # Here we would produce a result to Redis or a result queue
        
if __name__ == "__main__":
    process_task()