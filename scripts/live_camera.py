import cv2
import time
import uuid
import base64
import requests
import json

# --- Configuration ---
API_URL = "http://localhost:8000/ingest"

# OPTION A: Use Webcam (Index 0 is usually the default)
CAMERA_SOURCE = 0 

# OPTION B: Use a Video File (Uncomment line below and provide path)
# CAMERA_SOURCE = "/home/safhac/videos/test_gestures.mp4" 

def start_stream():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    # Verify camera opened
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {CAMERA_SOURCE}")
        return

    session_id = str(uuid.uuid4())
    print(f"--- Streaming Real Video Session: {session_id} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # 1. Resize to reduce network load (640x480 is standard for MediaPipe)
        frame = cv2.resize(frame, (640, 480))

        # 2. Encode Frame to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 3. Construct Payload
        payload = {
            "session_id": session_id,
            "metadata": {
                "source_id": "real-camera-01",
                "timestamp": time.time(),
                "resolution": "640x480",
                "encoding": "base64/jpeg"  # <--- ADDED: Required by server
            },
            "frame_data": jpg_as_text,
            "sensor_data": {                 # <--- ADDED: Required by server
                "camera_type": "webcam",
                "fps": 15
            } 
        }

        # 4. Send to API Gateway
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                print(f"✅ Sent Frame | Partition: {res.json().get('partition')}")
            else:
                print(f"⚠️ API Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")

        # Limit to ~15 FPS to prevent flooding while testing
        time.sleep(0.066) 

    cap.release()

if __name__ == "__main__":
    start_stream()