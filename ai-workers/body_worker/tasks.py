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

def detect_gesture(landmarks, frame_width, frame_height, frame):
    """
    Decodes 'HANDS_RAISED' and draws visual debug info on the frame.
    """
    # Landmarks: 11/12 (Shoulders), 13/14 (Elbows), 15/16 (Wrists)
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    # --- 1. THE LOGIC (Relaxed) ---
    # Hands are "raised" if wrists are higher (smaller Y) than elbows.
    hands_up = (
        left_wrist.y < left_elbow.y and 
        right_wrist.y < right_elbow.y
    )

    # --- 2. VISUAL DEBUGGING (Draw on the frame) ---
    # Convert normalized coordinates (0.0-1.0) to pixels
    lw_x, lw_y = int(left_wrist.x * frame_width), int(left_wrist.y * frame_height)
    rw_x, rw_y = int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)
    le_y_px = int(left_elbow.y * frame_height)

    # Draw Yellow Circles on Wrists
    cv2.circle(frame, (lw_x, lw_y), 15, (0, 255, 255), -1)
    cv2.circle(frame, (rw_x, rw_y), 15, (0, 255, 255), -1)

    # Draw Red Threshold Line (at Left Elbow height) to show the "Target"
    cv2.line(frame, (0, le_y_px), (frame_width, le_y_px), (0, 0, 255), 2)
    
    return "HANDS_RAISED" if hands_up else "NEUTRAL"

@shared_task(name='process_body_video')
def process_body_video(file_path, task_id):
    logger.info(f"[Body-Worker] Processing video: {file_path}")
    
    if not os.path.exists(file_path):
        return {"error": "File missing"}

    # 1. Setup Input Video
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 2. Setup Output Video Writer
    output_dir = "/app/media/results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{task_id}_labeled.mp4")
    
    # CRITICAL FIX: Use 'avc1' (H.264) for VS Code compatibility
    # If this fails, fallback to 'mp4v'
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    except Exception:
        logger.warning("avc1 codec failed, falling back to mp4v")
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
            
            # Prepare frame for MediaPipe
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            frame.flags.writeable = True
            
            current_gesture = "UNKNOWN"

            if results.pose_landmarks:
                # A. Draw Skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # B. Detect Gesture AND Draw Visual Debugging
                current_gesture = detect_gesture(
                    results.pose_landmarks.landmark, 
                    width, 
                    height, 
                    frame
                )

                # C. Draw Text Label
                color = (0, 255, 0) if current_gesture == "HANDS_RAISED" else (0, 0, 255)
                cv2.putText(
                    frame, 
                    f"Mode: {current_gesture}", 
                    (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, 
                    color, 
                    3, 
                    cv2.LINE_AA
                )

            out.write(frame)
            results_data.append({"frame": frame_count, "gesture": current_gesture})

    cap.release()
    out.release()

    json_path = os.path.join(output_dir, f"{task_id}.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Processing complete. Saved to {output_video_path}")
    return {"status": "success", "video_path": output_video_path, "json_path": json_path}