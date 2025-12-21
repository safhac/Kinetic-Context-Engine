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
    # 3. mjpg (MJPEG): Universal but produces huge files.
    codecs_to_try = ['avc1', 'mp4v', 'mjpg']

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
    Returns: str (Gesture Name)
    """
    # Landmarks: 13/14 (Elbows), 15/16 (Wrists)
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    # --- LOGIC: Wrist higher (smaller Y) than Elbow ---
    hands_up = (
        left_wrist.y < left_elbow.y and 
        right_wrist.y < right_elbow.y
    )

    # --- VISUAL DEBUGGING ---
    # Convert normalized coordinates (0.0-1.0) to pixels
    lw_x, lw_y = int(left_wrist.x * frame_width), int(left_wrist.y * frame_height)
    rw_x, rw_y = int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)
    le_y_px = int(left_elbow.y * frame_height)

    # 1. Draw Yellow Circles on Wrists
    cv2.circle(frame, (lw_x, lw_y), 15, (0, 255, 255), -1)
    cv2.circle(frame, (rw_x, rw_y), 15, (0, 255, 255), -1)

    # 2. Draw Red Threshold Line (at Left Elbow height)
    # This shows visually where the wrist needs to cross to be "UP"
    cv2.line(frame, (0, le_y_px), (frame_width, le_y_px), (0, 0, 255), 2)
    
    return "HANDS_RAISED" if hands_up else "NEUTRAL"

@shared_task(name='process_body_video', bind=True)
def process_body_video(self, file_path, task_id):
    """
    Main Celery Task.
    bind=True allows us to access 'self' for future state updates or error bubbling.
    """
    logger.info(f"[Body-Worker] Processing video: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File missing: {file_path}")
        # In the future, you can raise an exception here to bubble it up
        return {"status": "failed", "error": "File missing"}

    # 1. Setup Input Video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error("Could not open input video.")
        return {"status": "failed", "error": "Invalid video file"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 2. Setup Output Paths
    output_dir = "/app/media/results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{task_id}_labeled.mp4")
    
    # 3. Initialize Video Writer (Using the new Adapter)
    out = create_video_writer(output_video_path, fps, width, height)
    
    if not out:
        logger.critical("FATAL: Could not initialize ANY video writer. Check FFMPEG installation.")
        cap.release()
        # Bubbling error example: raise RuntimeError("VideoWriter initialization failed")
        return {"status": "failed", "error": "VideoWriter init failed"}

    frame_count = 0
    results_data = []

    try:
        # 4. Start MediaPipe Context
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Prepare frame (MediaPipe requires RGB)
                frame.flags.writeable = False
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame.flags.writeable = True
                
                current_gesture = "UNKNOWN"

                if results.pose_landmarks:
                    # A. Draw Standard Skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # B. Detect Gesture & Draw Debug Overlays
                    current_gesture = detect_and_draw_gesture(
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

                # Write frame to output video
                out.write(frame)
                results_data.append({"frame": frame_count, "gesture": current_gesture})

    except Exception as e:
        logger.error(f"Processing crashed at frame {frame_count}: {e}")
        # Re-raising allows Celery to mark task as FAILURE and retry if configured
        raise e 
        
    finally:
        # 5. Cleanup Resources
        cap.release()
        out.release()

    # 6. Save JSON Results
    json_path = os.path.join(output_dir, f"{task_id}.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Processing complete. Video saved to {output_video_path}")
    return {"status": "success", "video_path": output_video_path, "json_path": json_path}