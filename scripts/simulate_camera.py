import requests
import time
import random
import uuid
from datetime import datetime

# The API Gateway (Ingestion Service) URL
URL = "http://localhost:8000/ingest"

def generate_telemetry(session_id):
    """Generates fake camera data matching your schemas.py"""
    return {
        "session_id": session_id,
        "metadata": {
            "source_id": "camera-01",
            "timestamp": time.time(),
            "resolution": "1920x1080",
            "encoding": "h264"
        },
        "sensor_data": {
            "accelerometer": {
                "x": random.uniform(-1, 1),
                "y": random.uniform(-1, 1),
                "z": random.uniform(9.0, 10.0) # Gravity
            },
            "gyroscope": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            }
        }
    }

def start_camera():
    session_id = str(uuid.uuid4())
    print(f"--- Starting Recording Session: {session_id} ---")
    
    try:
        # Simulate 100 frames (10 seconds of video @ 10fps)
        for i in range(100):
            payload = generate_telemetry(session_id)
            
            # Send the HTTP POST to your Ingestion Service
            response = requests.post(URL, json=payload)
            
            if response.status_code == 202:
                print(f"Frame {i}: Sent -> Ingested (Partition {response.json().get('partition')})")
            else:
                print(f"Frame {i}: Failed ({response.status_code}) - {response.text}")
                
            # Sleep 0.1s to simulate real-time recording
            time.sleep(0.1)
            
    except requests.exceptions.ConnectionError:
        print("[Error] Could not connect to Ingestion Service. Is it running on port 8000?")

if __name__ == "__main__":
    start_camera()