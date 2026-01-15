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
    Tries multiple codecs to find one that works (H.264 -> MPEG-4).
    """
    codecs_to_try = ['avc1', 'mp4v']
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"VideoWriter initialized with codec: {codec}")
                return out
        except Exception as e:
            logger.warning(f"Codec {codec} init failed: {e}")
    return None


def detect_and_draw_gesture(landmarks, width, height, frame):
    """
    Decodes 'HANDS_RAISED' and draws visual debug info on the frame.
    """
    l_elbow, r_elbow = landmarks[13], landmarks[14]
    l_wrist, r_wrist = landmarks[15], landmarks[16]

    # Logic: Wrists higher (smaller Y) than Elbows
    hands_up = (l_wrist.y < l_elbow.y and r_wrist.y < r_elbow.y)
    gesture = "HANDS_RAISED" if hands_up else "NEUTRAL"

    # Visual Debugging (Yellow Wrists & Red Threshold Line)
    lw_pos = (int(l_wrist.x * width), int(l_wrist.y * height))
    rw_pos = (int(r_wrist.x * width), int(r_wrist.y * height))
    threshold_y = int(l_elbow.y * height)

    cv2.circle(frame, lw_pos, 15, (0, 255, 255), -1)
    cv2.circle(frame, rw_pos, 15, (0, 255, 255), -1)
    cv2.line(frame, (0, threshold_y), (width, threshold_y), (0, 0, 255), 2)

    return gesture


@shared_task(name='process_body_video', bind=True)
def process_body_video(self, file_path, task_id):
    logger.info(f"[Body-Worker] Processing: {file_path}")

    # 1. Validation & Setup
    if not os.path.exists(file_path):
        return {"status": "failed", "error": "File missing"}

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {"status": "failed", "error": "Invalid video"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir = "/app/media/results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{task_id}_labeled.mp4")

    out = create_video_writer(output_video_path, fps, width, height)
    if not out:
        cap.release()
        return {"status": "failed", "error": "VideoWriter init failed"}

    # 2. Responsive Layout Setup (Calculate Once)
    # Scales font size based on video width (e.g., 4K video gets larger text)
    font_scale = (width / 1000.0) * 1.5
    thickness = int((width / 1000.0) * 3) + 1
    x_pos = int(width * 0.05)  # 5% padding from left
    y_pos = int(height * 0.1)  # 10% padding from top

    frame_count = 0
    results_data = []

    # 3. Processing Loop
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Prepare MediaPipe Input
                frame.flags.writeable = False
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame.flags.writeable = True

                current_gesture = "UNKNOWN"

                if results.pose_landmarks:
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # Detect Gesture
                    current_gesture = detect_and_draw_gesture(
                        results.pose_landmarks.landmark, width, height, frame)

                # Draw Status Text (Updates every frame)
                color = (0, 255, 0) if current_gesture == "HANDS_RAISED" else (
                    0, 0, 255)
                cv2.putText(frame, f"Mode: {current_gesture}", (x_pos, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

                out.write(frame)
                results_data.append(
                    {"frame": frame_count, "gesture": current_gesture})

    except Exception as e:
        logger.error(f"Processing crashed: {e}")
        raise e
    finally:
        cap.release()
        out.release()

    # 4. Save JSON Results
    json_path = os.path.join(output_dir, f"{task_id}.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Complete. Saved to {output_video_path}")
    return {"status": "success", "video_path": output_video_path, "json_path": json_path}
