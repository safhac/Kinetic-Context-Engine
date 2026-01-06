import math
import numpy as np
import cv2

# --- State Storage ---
_face_history = {
    "nose_y": [],
    "cheek_redness": []
}

# --- Main Signal Aggregator ---


def get_active_face_signals(face_landmarks, pose_landmarks=None, frame=None):
    """
    Checks ALL implemented signals against the current frame.
    Args:
        face_landmarks: MediaPipe Face Mesh landmarks.
        pose_landmarks: MediaPipe Pose landmarks (optional).
        frame: The BGR image frame (optional, required for redness/flushing).
    """
    signals = []

    # --- Basic Geometry ---
# ... [Keep all existing geometric checks: chin_thrust, etc.] ...
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
    if detect_happiness_genuine(face_landmarks):
        signals.append("happiness_genuine")
    if detect_surprise_genuine(face_landmarks):
        signals.append("surprise_genuine")
    if detect_fear(face_landmarks):
        signals.append("fear")
    if detect_confirmation_glance(face_landmarks):
        signals.append("confirmation_glance")
    if detect_vertical_head_shake(face_landmarks):
        signals.append("nodding_yes")
    if frame is not None:
        if detect_flushing(frame, face_landmarks):
            signals.append("flushing_redness")

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


# --- Core Logic / Motion Tracking ---

def detect_oscillating_vertical_movement(landmark, window_frames=30, min_cycles=2):
    """
    Detects simple harmonic motion (nodding) by counting direction changes.
    """
    history = _face_history["nose_y"]
    history.append(landmark.y)

    # Keep strictly the last N frames
    if len(history) > window_frames:
        history.pop(0)

    # Need enough data to detect oscillation
    if len(history) < window_frames:
        return False

    # Calculate Velocities
    velocities = [history[i] - history[i-1] for i in range(1, len(history))]

    # Count Zero Crossings (Direction Changes)
    flips = 0
    for i in range(1, len(velocities)):
        if (velocities[i] > 0 and velocities[i-1] < 0) or \
           (velocities[i] < 0 and velocities[i-1] > 0):
            # Filter out tiny jitters (noise)
            if abs(velocities[i]) > 0.002:
                flips += 1

    return flips >= (min_cycles * 2)


def check_redness_spike(frame, cheek_landmarks, face_landmarks):
    """
    Detects sudden increase in red channel intensity in cheek regions.
    """
    h, w, _ = frame.shape
    redness_values = []

    for idx in cheek_landmarks:
        lm = face_landmarks.landmark[idx]
        px, py = int(lm.x * w), int(lm.y * h)

        # Safety check for image bounds
        if py < 5 or py > h-5 or px < 5 or px > w-5:
            continue

        roi = frame[py-5:py+5, px-5:px+5]
        if roi.size == 0:
            continue

        mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
        if mean_g > 0:
            redness_values.append(mean_r / mean_g)

    if not redness_values:
        return False

    avg_redness = sum(redness_values) / len(redness_values)

    # History Check
    history = _face_history["cheek_redness"]
    history.append(avg_redness)
    if len(history) > 60:  # ~2 seconds at 30fps
        history.pop(0)

    # Logic: Is current redness 10% higher than average of last 2 seconds?
    baseline = sum(history) / len(history)
    return avg_redness > (baseline * 1.10)


def calculate_blink_rate(upper_lid, lower_lid, current_rate):
    if abs(upper_lid.y - lower_lid.y) < 0.005:
        return current_rate + 1
    return current_rate


# --- Eye & Iris Logic ---

def detect_iris_center_ratio(face_landmarks, mode="horizontal"):
    """
    Calculates the position of the iris relative to the eye corners or lids.
    """
    lm = face_landmarks.landmark
    iris_center = lm[468]

    if mode == "horizontal":
        inner_corner = lm[33]
        outer_corner = lm[133]
        eye_width = abs(outer_corner.x - inner_corner.x)
        if eye_width == 0:
            return 0.5
        dist_to_inner = abs(iris_center.x - inner_corner.x)
        return dist_to_inner / eye_width

    elif mode == "vertical":
        # 159 = Top Lid, 145 = Bottom Lid
        eye_height = abs(lm[159].y - lm[145].y)
        if eye_height == 0:
            return 0.5
        dist_to_top = abs(iris_center.y - lm[159].y)
        return dist_to_top / eye_height

    return 0.5


def detect_confirmation_glance(face_landmarks):
    # Detects a quick lateral shift of pupils
    ratio = detect_iris_center_ratio(face_landmarks, mode="horizontal")
    return ratio < 0.3 or ratio > 0.7


def detect_eye_squint(face_landmarks):
    # Detects narrowing of the eyelids
    lm = face_landmarks.landmark
    left_open = abs(lm[159].y - lm[145].y)
    right_open = abs(lm[386].y - lm[374].y)
    face_height = abs(lm[10].y - lm[152].y)

    if face_height == 0:
        return False

    avg_ratio = ((left_open + right_open) / 2) / face_height
    # Threshold: Open but Small (squinting vs blinking)
    return 0.005 < avg_ratio < 0.03


# --- Geometric Signal Implementations ---

def detect_chin_thrust(face_landmarks):
    # Chin (152) closer to camera than Nose (1) significantly
    return face_landmarks.landmark[152].z < (face_landmarks.landmark[1].z - 0.05)


def detect_lip_compression(face_landmarks):
    # Lips (13, 14) are pressed tight
    return abs(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y) < 0.005


def detect_head_tilt(face_landmarks):
    # Eyes (33, 263) are not horizontal
    return abs(face_landmarks.landmark[33].y - face_landmarks.landmark[263].y) > 0.05


def detect_eyebrow_flash(face_landmarks):
    # Distance between eyebrow top (105) and eye (33)
    return abs(face_landmarks.landmark[105].y - face_landmarks.landmark[33].y) > 0.05


def detect_eyebrow_narrowing(face_landmarks):
    # Measures distance between inner eyebrows
    lm = face_landmarks.landmark
    return abs(lm[55].x - lm[285].x) < 0.02


def detect_head_downcast_face_only(face_landmarks):
    # Nose (1) significantly below eye line
    mid_eye_y = (face_landmarks.landmark[33].y +
                 face_landmarks.landmark[263].y) / 2
    return (face_landmarks.landmark[1].y - mid_eye_y) > 0.15


def detect_head_downcast_with_pose(face_landmarks, pose_landmarks):
    # More robust version if pose is available.
    # Checks nose relative to shoulder line.
    nose = face_landmarks.landmark[1]
    shoulders_y = (
        pose_landmarks.landmark[11].y + pose_landmarks.landmark[12].y) / 2
    return nose.y > (shoulders_y - 0.1)


def detect_vertical_head_shake(face_landmarks):
    # Monitors pitch of the nose for 'Yes' motion
    return detect_oscillating_vertical_movement(face_landmarks.landmark[1])


def measure_facial_compression(face_landmarks):
    # Detects 'scrunching' (nose/lip gap shrinks)
    lm = face_landmarks.landmark
    scrunch_dist = _dist(lm[1], lm[13])
    face_height = _dist(lm[10], lm[152])

    if face_height == 0:
        return False
    return (scrunch_dist / face_height) < 0.12


def detect_disgust(face_landmarks):
    return measure_facial_compression(face_landmarks)


def detect_teeth_sucking(face_landmarks):
    # Detects lateral lip tension
    lm = face_landmarks.landmark
    width = abs(lm[61].x - lm[291].x)
    return width > 0.08


def detect_lip_biting(face_landmarks):
    # Detects lips passing teeth barrier (Z-depth check)
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    return abs(upper_lip.z - lower_lip.z) > 0.01


def detect_jaw_clenching(face_landmarks):
    # Monitors muscular tension in temple/jaw
    lm = face_landmarks.landmark
    dist = abs(lm[21].y - lm[172].y)
    return dist < 0.05


def detect_nostril_dilation(face_landmarks):
    # Measures horizontal distance between nostril wings
    lm = face_landmarks.landmark
    width = abs(lm[64].x - lm[294].x)
    return width > 0.04


def detect_yawn(face_landmarks):
    # Measures vertical jaw gap
    lm = face_landmarks.landmark
    gap = abs(lm[13].y - lm[14].y)
    return gap > 0.15


def detect_flushing(frame, face_landmarks):
    # Analyzes color space in cheek regions (205, 425)
    return check_redness_spike(frame, [205, 425], face_landmarks)


# --- Complex Emotional Signals ---

def detect_happiness_genuine(face_landmarks):
    # Duchenne smile: Lip pull + Eye crinkle
    lm = face_landmarks.landmark
    lip_width = abs(lm[61].x - lm[291].x)
    eye_squint = abs(lm[159].y - lm[145].y)
    return lip_width > 0.07 and eye_squint < 0.01


def detect_surprise_genuine(face_landmarks):
    # Eyebrows raised + Jaw drop
    lm = face_landmarks.landmark
    eyebrow_dist = abs(lm[105].y - lm[33].y)
    jaw_gap = abs(lm[13].y - lm[14].y)
    return eyebrow_dist > 0.06 and jaw_gap > 0.08


def detect_fear(face_landmarks):
    # Eyes widen + Lips stretch
    lm = face_landmarks.landmark
    eye_sclera = abs(lm[159].y - lm[145].y)
    lip_stretch = abs(lm[61].x - lm[291].x)
    return eye_sclera > 0.05 and lip_stretch > 0.08


# --- Hand-to-Face Logic (Needs Pose) ---

def detect_head_support(face_landmarks, pose_landmarks):
    # Hand resting on chin/face
    chin = face_landmarks.landmark[152]
    # Using wrist/elbow approximation (13/14 are elbows, 15/16 are wrists in MediaPipe)
    # Checking generic proximity of arm to chin
    left_hand = pose_landmarks.landmark[13]
    right_hand = pose_landmarks.landmark[14]
    return _dist(chin, left_hand) < 0.1 or _dist(chin, right_hand) < 0.1


def detect_hushing(face_landmarks, pose_landmarks):
    # Hand covering mouth
    mouth = face_landmarks.landmark[13]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(mouth, l_hand) < 0.1 or _dist(mouth, r_hand) < 0.1


def detect_finger_to_nose(face_landmarks, pose_landmarks):
    # Concealed mouth cover
    nose = face_landmarks.landmark[1]
    l_index = pose_landmarks.landmark[19]
    r_index = pose_landmarks.landmark[20]
    return _dist(nose, l_index) < 0.05 or _dist(nose, r_index) < 0.05


def detect_eyelid_rubbing(face_landmarks, pose_landmarks):
    # Hand rubbing eyes
    eye = face_landmarks.landmark[33]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(eye, l_hand) < 0.1 or _dist(eye, r_hand) < 0.1


def detect_object_in_mouth(face_landmarks, pose_landmarks):
    # Detects hand/object near mouth (e.g. smoking/pen)
    mouth = face_landmarks.landmark[13]
    l_hand = pose_landmarks.landmark[15]
    r_hand = pose_landmarks.landmark[16]
    return _dist(mouth, l_hand) < 0.05 or _dist(mouth, r_hand) < 0.05
