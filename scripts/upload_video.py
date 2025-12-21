import cv2
import requests
import base64
import time
import uuid
import json
import sys

# --- CONFIGURATION ---
VIDEO_PATH = "sample.mp4"       # <--- PUT YOUR VIDEO FILENAME HERE
API_URL = "http://localhost:8000/ingest" # Check your docker-compose for the port (usually 8000 or 8080)
SESSION_ID = str(uuid.uuid4())

def upload_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🚀 Starting upload for Session: {SESSION_ID}")
    print(f"📼 Video: {video_path} ({total_frames} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Encode Frame to JPEG then Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 2. Build Payload (Matching your new Schema)
        payload = {
            "session_id": SESSION_ID,
            "metadata": {
                "source_id": "uploader_script",
                "timestamp": time.time(),
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "encoding": "jpg"
            },
            "sensor_data": {
                "accelerometer": {"x": 0, "y": 0, "z": 9.8}, # Mock data
                "gyroscope": {"x": 0, "y": 0, "z": 0}
            },
            "frame_data": jpg_as_text  # <--- The video data
        }

        # 3. Send to Ingestion Service
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"\r✅ Uploaded Frame {frame_count}/{total_frames}", end="")
            else:
                print(f"\n❌ Failed Frame {frame_count}: {response.text}")
        except Exception as e:
            print(f"\n❌ Connection Error: {e}")
            break

        frame_count += 1
        # Optional: Small sleep to prevent crashing your own PC if it sends too fast
        # time.sleep(0.01) 

    cap.release()
    print(f"\n\n✨ Upload Complete! Session ID: {SESSION_ID}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = VIDEO_PATH
    
    upload_video(video_file)