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

def create_video_writer(output_path, fps, width, height):
    """
    Tries multiple codecs to find one that works on the current system.
    Returns: (cv2.VideoWriter) or None if all fail.
    """
    # 1. avc1 (H.264): Best for browsers/VS Code.
    # 2. mp4v (MPEG-4): Robust fallback, works in VLC.
    codecs_to_try = ['avc1', 'mp4v']

    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                logger.info(f"VideoWriter initialized successfully using codec: {codec}")
                return out
            else:
                logger.warning(f"Codec {codec} failed to open writer.")
                
        except Exception as e:
            logger.warning(f"Codec {codec} crashed during init: {e}")

    return None

def detect_and_draw_gesture(landmarks, frame_width, frame_height, frame):
    """
    Decodes 'HANDS_RAISED' and draws visual debug info (circles/lines) on the frame.
    """
    # Landmarks: 13/14 (Elbows), 15/16 (Wrists)
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    # LOGIC: Hands are "up" if Wrist Y < Elbow Y (Remember Y=0 is top)
    hands_up = (
        left_wrist.y < left_elbow.y and 
        right_wrist.y < right_elbow.y
    )

    # VISUAL DEBUGGING
    lw_x, lw_y = int(left_wrist.x * frame_width), int(left_wrist.y * frame_height)
    rw_x, rw_y = int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)
    le_y_px = int(left_elbow.y * frame_height)

    # Draw Yellow Circles on Wrists
    cv2.circle(frame, (lw_x, lw_y), 15, (0, 255, 255), -1)
    cv2.circle(frame, (rw_x, rw_y), 15, (0, 255, 255), -1)

    # Draw Red Threshold Line (at Left Elbow height)
    cv2.line(frame, (0, le_y_px), (frame_width, le_y_px), (0, 0, 255), 2)
    
    return "HANDS_RAISED" if hands_up else "NEUTRAL"

@shared_task(name='process_body_video', bind=True)
def process_body_video(self, file_path, task_id):
    logger.info(f"[Body-Worker] Processing video: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File missing: {file_path}")
        return {"status": "failed", "error": "File missing"}

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error("Could not open input video.")
        return {"status": "failed", "error": "Invalid video file"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir = "/app/media/results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{task_id}_labeled.mp4")
    
    # Initialize Video Writer
    out = create_video_writer(output_video_path, fps, width, height)
    
    if not out:
        logger.critical("FATAL: Could not initialize ANY video writer.")
        cap.release()
        return {"status": "failed", "error": "VideoWriter init failed"}

    frame_count = 0
    results_data = []

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Prepare frame
                frame.flags.writeable = False
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame.flags.writeable = True
                
                current_gesture = "UNKNOWN"

                if results.pose_landmarks:
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Detect & Draw Logic
                    current_gesture = detect_and_draw_gesture(
                        results.pose_landmarks.landmark, 
                        width, 
                        height, 
                        frame
                    )

                    # Draw Text
                    color = (0, 255, 0) if current_gesture == "HANDS_RAISED" else (0, 0, 255)
                    cv2.putText(frame, f"Mode: {current_gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

                out.write(frame)
                results_data.append({"frame": frame_count, "gesture": current_gesture})

    except Exception as e:
        logger.error(f"Processing crashed: {e}")
        raise e 
        
    finally:
        cap.release()
        out.release()

    json_path = os.path.join(output_dir, f"{task_id}.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Processing complete. Saved to {output_video_path}")
    return {"status": "success", "video_path": output_video_path, "json_path": json_path}