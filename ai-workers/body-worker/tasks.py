# ai-workers/body-worker/tasks.py
import cv2
import os
import mediapipe as mp
from main import celery_app # Import the app from the parent directory

# Setup MediaPipe once (global scope) to save overhead
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

@celery_app.task(name='process_body_video')
def process_body_video(file_path, task_id):
    """
    This function is triggered automatically by Redis.
    It does NOT need a while True loop.
    """
    print(f"[Body-Worker] Received task {task_id} for file: {file_path}")
    
    results_data = []
    
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Simple example: Store nose coordinates if found
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                results_data.append({
                    "frame": frame_count, 
                    "nose_x": nose.x, 
                    "nose_y": nose.y
                })
            
            frame_count += 1
            
        cap.release()
        
        # --- Save Results ---
        result_path = f"/app/media/results/{task_id}.json"
        with open(result_path, "w") as f:
            import json
            json.dump(results_data, f)
            
        print(f"[Body-Worker] Finished. Saved to {result_path}")
        
    except Exception as e:
        print(f"[Body-Worker] Error: {e}")
        return {"status": "failed", "error": str(e)}
        
    finally:
        # --- Cleanup: Delete the video to save space ---
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"status": "success", "task_id": task_id}