import time
import os
import json
import cv2
import mediapipe as mp
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# --- Configuration ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_gesture(landmarks):
    """
    Simple logic to detect 'HANDS_RAISED'
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    # Y-coordinate increases downwards. Smaller Y = Higher up.
    hands_above_shoulders = (
        left_wrist.y < left_shoulder.y and 
        right_wrist.y < right_shoulder.y
    )

    if hands_above_shoulders:
        return "HANDS_RAISED"
    return "NEUTRAL"

@shared_task(name='process_body_video')
def process_body_video(file_path, task_id):
    logger.info(f"[Body-Worker] Processing video: {file_path}")
    
    if not os.path.exists(file_path):
        return {"error": "File missing"}

    # 1. Setup Input Video
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # Default to 30 if unknown

    # 2. Setup Output Video Writer
    output_dir = "/app/media/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # We create a new file with '_labeled.mp4' suffix
    output_video_path = os.path.join(output_dir, f"{task_id}_labeled.mp4")
    
    # 'mp4v' is a widely supported codec for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    results_data = []

    # 3. Start Processing
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Optimization: Mark image as not writeable to pass by reference
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Re-enable drawing
            frame.flags.writeable = True
            
            current_gesture = "UNKNOWN"

            if results.pose_landmarks:
                # A. Decode Gesture
                current_gesture = detect_gesture(results.pose_landmarks.landmark)
                
                # B. Draw Skeleton (The "Stick Figure")
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # C. Draw Gesture Text (The "Label")
                # Color: Green (0,255,0) if raised, Blue (255,0,0) if neutral
                color = (0, 255, 0) if current_gesture == "HANDS_RAISED" else (255, 0, 0)
                
                cv2.putText(
                    frame, 
                    f"Gesture: {current_gesture}", 
                    (50, 50),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5,       # Font Scale
                    color, 
                    3,         # Thickness
                    cv2.LINE_AA
                )

            # D. Write the annotated frame to the new video file
            out.write(frame)

            # Save metadata for JSON
            results_data.append({
                "frame": frame_count,
                "gesture": current_gesture
            })

    # 4. Cleanup
    cap.release()
    out.release()

    # 5. Save JSON report as well
    json_path = os.path.join(output_dir, f"{task_id}.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Processing complete. Video saved to {output_video_path}")
    return {"status": "success", "video_path": output_video_path, "json_path": json_path}