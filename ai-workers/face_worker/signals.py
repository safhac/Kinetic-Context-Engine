import math
import numpy as np
import cv2

# State storage for Face Worker
_face_history = {
    "nose_y": [],
    "cheek_redness": []
}

# --- Main Entry Point ---


def get_active_face_signals(face_landmarks, pose_landmarks=None):
    """
    Checks ALL implemented signals against the current frame.
    Accepts optional pose_landmarks for hand-to-face gestures.
    """
    signals = []

    # --- Basic Geometry ---
    if detect_chin_thrust(face_landmarks):
        signals.append("chin_thrust")
    if detect_lip_compression(face_landmarks):
        signals.append("lips_compressed")
    if detect_head_tilt(face_landmarks):
        signals.append("head_tilted")
    if detect_eyebrow_flash(face_landmarks):
        signals.append("eyebrow_flash")
    if detect_head_downcast_face_only(face_landmarks):
        signals.append("head_down")
    if detect_disgust(face_landmarks):
        signals.append("disgust")
    if detect_eye_squint(face_landmarks):
        signals.append("eye_squint")
    if detect_teeth_sucking(face_landmarks):
        signals.append("teeth_sucking")
    if detect_jaw_clenching(face_landmarks):
        signals.append("jaw_clenching")
    if detect_nostril_dilation(face_landmarks):
        signals.append("nostril_dilation")
    if detect_yawn(face_landmarks):
        signals.append("yawn")
    if detect_eyebrow_narrowing(face_landmarks):
        signals.append("eyebrow_narrowing")

    # --- Complex Expressions ---
    if detect_happiness_genuine(face_landmarks):
        signals.append("happiness_genuine")
    if detect_surprise_genuine(face_landmarks):
        signals.append("surprise_genuine")
    if detect_fear(face_landmarks):
        signals.append("fear")
    if detect_confirmation_glance(face_landmarks):
        signals.append("confirmation_glance")

    # --- Motion Based ---
    if detect_vertical_head_shake(face_landmarks):
        signals.append("nodding_yes")

    # --- Hand-to-Face (Requires Pose) ---
    if pose_landmarks:
        if detect_head_support(face_landmarks, pose_landmarks):
            signals.append("head_support")
        if detect_hushing(face_landmarks, pose_landmarks):
            signals.append("hushing")
        if detect_finger_to_nose(face_landmarks, pose_landmarks):
            signals.append("finger_to_nose")
        if detect_eyelid_rubbing(face_landmarks, pose_landmarks):
            signals.append("eyelid_rubbing")
        if detect_object_in_mouth(face_landmarks, pose_landmarks):
            signals.append("object_in_mouth")

    return signals

# --- Helpers ---


def _dist(p1, p2):
    """Euclidean distance ignoring Z depth"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def detect_oscillating_vertical_movement(landmark, window_frames=30, min_cycles=2):
    history = _face_history["nose_y"]
    history.append(landmark.y)
    if len(history) > window_frames:
        history.pop(0)
    if len(history) < window_frames:
        return False

    velocities = [history[i] - history[i-1] for i in range(1, len(history))]
    flips = 0
    for i in range(1, len(velocities)):
        if (velocities[i] > 0 and velocities[i-1] < 0) or \
           (velocities[i] < 0 and velocities[i-1] > 0):
            if abs(velocities[i]) > 0.002:
                flips += 1
    return flips >= (min_cycles * 2)


def detect_iris_center_ratio(face_landmarks, mode="horizontal"):
    lm = face_landmarks.landmark
    iris_center = lm[468]
    inner_corner = lm[33]
    outer_corner = lm[133]

    if mode == "horizontal":
        eye_width = abs(outer_corner.x - inner_corner.x)
        if eye_width == 0:
            return 0.5
        dist_to_inner = abs(iris_center.x - inner_corner.x)
        return dist_to_inner / eye_width
    return 0.5

# --- Signal Implementations ---


def detect_chin_thrust(face_landmarks):
    # [cite_start]Chin (152) closer to camera than Nose (1) significantly [cite: 40]
    return face_landmarks.landmark[152].z < (face_landmarks.landmark[1].z - 0.05)


def detect_lip_compression(face_landmarks):
    # [cite_start]Lips (13, 14) are pressed tight [cite: 58]
    return abs(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y) < 0.005


def detect_head_tilt(face_landmarks):
    # [cite_start]Eyes (33, 263) are not horizontal [cite: 18]
    return abs(face_landmarks.landmark[33].y - face_landmarks.landmark[263].y) > 0.05


def detect_eyebrow_flash(face_landmarks):
    # [cite_start]Distance between eyebrow top (105) and eye (33) [cite: 47, 51]
    return abs(face_landmarks.landmark[105].y - face_landmarks.landmark[33].y) > 0.05


def detect_head_downcast_face_only(face_landmarks):
    # [cite_start]Nose (1) significantly below eye line [cite: 54]
    mid_eye_y = (face_landmarks.landmark[33].y +
                 face_landmarks.landmark[263].y) / 2
    return (face_landmarks.landmark[1].y - mid_eye_y) > 0.15


def detect_vertical_head_shake(face_landmarks):
    [cite_start]  # [cite: 147]
    return detect_oscillating_vertical_movement(face_landmarks.landmark[1])


def detect_disgust(face_landmarks):
    [cite_start]  # [cite: 31]
    return measure_facial_compression(face_landmarks)


def measure_facial_compression(face_landmarks):
    lm = face_landmarks.landmark
    scrunch_dist = _dist(lm[1], lm[13])
    face_height = _dist(lm[10], lm[152])
    if face_height == 0:
        return False
    return (scrunch_dist / face_height) < 0.12


def detect_eye_squint(face_landmarks):
    [cite_start]  # [cite: 179]
    lm = face_landmarks.landmark
    left_open = abs(lm[159].y - lm[145].y)
    right_open = abs(lm[386].y - lm[374].y)
    face_height = abs(lm[10].y - lm[152].y)
    if face_height == 0:
        return False
    avg_ratio = ((left_open + right_open) / 2) / face_height
    return 0.005 < avg_ratio < 0.03


def detect_confirmation_glance(face_landmarks):
    [cite_start]  # [cite: 89]
    ratio = detect_iris_center_ratio(face_landmarks, mode="horizontal")
    return ratio < 0.3 or ratio > 0.7


def detect_teeth_sucking(face_landmarks):
    [cite_start]  # [cite: 65, 67]
    lm = face_landmarks.landmark
    width = abs(lm[61].x - lm[291].x)
    return width > 0.08  # Threshold tuned for lateral stretch


def detect_jaw_clenching(face_landmarks):
    [cite_start]  # [cite: 80, 81]
    lm = face_landmarks.landmark
    dist = abs(lm[21].y - lm[172].y)
    return dist < 0.05


def detect_nostril_dilation(face_landmarks):
    [cite_start]  # [cite: 82]
    lm = face_landmarks.landmark
    width = abs(lm[64].x - lm[294].x)
    return width > 0.04


def detect_yawn(face_landmarks):
    [cite_start]  # [cite: 92]
    lm = face_landmarks.landmark
    gap = abs(lm[13].y - lm[14].y)
    return gap > 0.15


def detect_happiness_genuine(face_landmarks):
    [cite_start]  # [cite: 98]
    lm = face_landmarks.landmark
    lip_width = abs(lm[61].x - lm[291].x)
    eye_squint = abs(lm[159].y - lm[145].y)
    return lip_width > 0.07 and eye_squint < 0.01


def detect_surprise_genuine(face_landmarks):
    [cite_start]  # [cite: 157]
    lm = face_landmarks.landmark
    eyebrow_dist = abs(lm[105].y - lm[33].y)
    jaw_gap = abs(lm[13].y - lm[14].y)
    return eyebrow_dist > 0.06 and jaw_gap > 0.08


def detect_fear(face_landmarks):
    [cite_start]  # [cite: 199]
    lm = face_landmarks.landmark
    eye_sclera = abs(lm[159].y - lm[145].y)
    lip_stretch = abs(lm[61].x - lm[291].x)
    return eye_sclera > 0.05 and lip_stretch > 0.08


def detect_eyebrow_narrowing(face_landmarks):
    [cite_start]  # [cite: 119]
    lm = face_landmarks.landmark
    return abs(lm[55].x - lm[285].x) < 0.02

# --- Hand-to-Face Logic (Needs Pose) ---


def detect_head_support(face_landmarks, pose_landmarks):
    [cite_start]  # [cite: 150]
    chin = face_landmarks.landmark[152]
    # Using wrist/elbow area approximation
    left_hand = pose_landmarks.landmark[13]
    right_hand = pose_landmarks.landmark[14]
    # Check proximity
    return _dist(chin, left_hand) < 0.1 or _dist(chin, right_hand) < 0.1


def detect_hushing(face_landmarks, pose_landmarks):
    [cite_start]  # [cite: 398]
    mouth = face_landmarks.landmark[13]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(mouth, l_hand) < 0.1 or _dist(mouth, r_hand) < 0.1


def detect_finger_to_nose(face_landmarks, pose_landmarks):
    [cite_start]  # [cite: 472]
    nose = face_landmarks.landmark[1]
    # Check index fingers (19, 20)
    l_index = pose_landmarks.landmark[19]
    r_index = pose_landmarks.landmark[20]
    return _dist(nose, l_index) < 0.05 or _dist(nose, r_index) < 0.05


def detect_eyelid_rubbing(face_landmarks, pose_landmarks):
    [cite_start]  # [cite: 463]
    eye = face_landmarks.landmark[33]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(eye, l_hand) < 0.1 or _dist(eye, r_hand) < 0.1


def detect_object_in_mouth(face_landmarks, pose_landmarks):
    [cite_start]  # [cite: 74]
    mouth = face_landmarks.landmark[13]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(mouth, l_hand) < 0.05 or _dist(mouth, r_hand) < 0.05
