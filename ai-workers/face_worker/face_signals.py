import math
import numpy as np
import cv2

# State storage for Face Worker
_face_history = {
    "nose_y": [],
    "cheek_redness": []  # Stores timestamped Y positions
}


def detect_oscillating_vertical_movement(landmark, window_frames=30, min_cycles=2):
    """
    Detects simple harmonic motion (nodding) by counting direction changes.
    Args:
        landmark: The MediaPipe landmark (e.g., Nose tip).
        window_frames: How many past frames to analyze (30 frames ~ 1 second).
        min_cycles: How many Up/Down flips constitute a "nod".
    """
    # 1. Update History
    history = _face_history["nose_y"]
    history.append(landmark.y)

    # Keep strictly the last N frames
    if len(history) > window_frames:
        history.pop(0)

    # Need enough data to detect oscillation
    if len(history) < window_frames:
        return False

    # 2. Calculate Velocities (Delta between frames)
    # v[i] = position[i] - position[i-1]
    velocities = []
    for i in range(1, len(history)):
        velocities.append(history[i] - history[i-1])

    # 3. Count Zero Crossings (Direction Changes)
    # A change happens when velocity goes from + to - or - to +
    flips = 0
    for i in range(1, len(velocities)):
        if (velocities[i] > 0 and velocities[i-1] < 0) or \
           (velocities[i] < 0 and velocities[i-1] > 0):
            # Filter out tiny jitters (noise)
            if abs(velocities[i]) > 0.002:  # Threshold 0.2% of screen
                flips += 1

    # A full cycle (Down-Up) is 2 flips. A "Nod" is typically at least 1.5 cycles.
    return flips >= (min_cycles * 2)


def detect_vertical_head_shake(face_landmarks):
    # Monitors pitch of the nose (1) for 'Yes' motion
    return detect_oscillating_vertical_movement(face_landmarks.landmark[1])


def get_active_face_signals(face_landmarks):
    signals = []

    # Chin Thrust [cite: 40]
    if detect_chin_thrust(face_landmarks):
        signals.append("chin_thrust")

    # Lip Compression [cite: 58]
    if detect_lip_compression(face_landmarks):
        signals.append("lips_compressed")

    # Head Tilt [cite: 18]
    if detect_head_tilt(face_landmarks):
        signals.append("head_tilted")

    return signals


def detect_eyebrow_flash(face_landmarks):
    # Distance between eyebrow top and eye center
    eyebrow = face_landmarks.landmark[105]
    eye = face_landmarks.landmark[33]
    dist = abs(eyebrow.y - eye.y)
    return dist > 0.04  # Threshold for 'raised' state


def detect_head_downcast(face_landmarks, pose_landmarks):
    nose = face_landmarks.landmark[1]
    shoulders_y = (
        pose_landmarks.landmark[11].y + pose_landmarks.landmark[12].y) / 2
    return nose.y > (shoulders_y - 0.1)  # Head is low relative to shoulders

# Simplified EAR logic


def detect_iris_center_ratio(face_landmarks, mode="horizontal"):
    """
    Calculates the position of the iris relative to the eye corners.
    Used for 'Confirmation Glance'  and 'Facing' logic.

    Returns:
        0.5 = Centered (Looking straight)
        0.0 = Looking far Left (Subject's right)
        1.0 = Looking far Right (Subject's left)
    """
    lm = face_landmarks.landmark

    # Left Eye Landmarks (MediaPipe Face Mesh)
    # 468 = Iris Center
    # 33 = Inner Corner, 133 = Outer Corner
    iris_center = lm[468]
    inner_corner = lm[33]
    outer_corner = lm[133]

    if mode == "horizontal":
        # Calculate eye width
        eye_width = abs(outer_corner.x - inner_corner.x)
        if eye_width == 0:
            return 0.5

        # Calculate distance of iris from inner corner
        dist_to_inner = abs(iris_center.x - inner_corner.x)

        # Ratio: 0 (Inner) to 1 (Outer)
        ratio = dist_to_inner / eye_width
        return ratio

    elif mode == "vertical":
        # Useful for 'Head Downcast' validation or rolling eyes
        # 159 = Top Lid, 145 = Bottom Lid
        eye_height = abs(lm[159].y - lm[145].y)
        if eye_height == 0:
            return 0.5

        dist_to_top = abs(iris_center.y - lm[159].y)
        return dist_to_top / eye_height

    return 0.5


def measure_facial_compression(face_landmarks):
    """
    Detects the 'scrunching' of the nose and upper lip typical of disgust.
    """
    lm = face_landmarks.landmark

    # 1. Define Key Points
    nose_tip = lm[1]
    upper_lip_top = lm[13]
    chin = lm[152]
    forehead = lm[10]  # Top of head

    # 2. Calculate Distances
    # The "Scrunch Zone": Distance between nose and lip
    scrunch_dist = _dist(nose_tip, upper_lip_top)

    # The "Face Scale": Total face height (to normalize for distance from camera)
    face_height = _dist(forehead, chin)

    # Safety check
    if face_height == 0:
        return False

    # 3. Calculate Ratio
    # Normal ratio is usually around 0.15 - 0.20
    # A "Scrunched" face pulls the lip up, shrinking this distance.
    ratio = scrunch_dist / face_height

    # Threshold: If the nose-to-lip gap is less than 12% of total face height
    return ratio < 0.12


def check_redness_spike(frame, cheek_landmarks, face_landmarks):
    """
    Detects sudden increase in red channel intensity in cheek regions.
    Args:
        frame: The BGR numpy image.
        cheek_landmarks: List of landmark indices (e.g., [205, 425]).
        face_landmarks: The full landmark object (to get coordinates).
    """
    h, w, _ = frame.shape
    redness_values = []

    for idx in cheek_landmarks:
        lm = face_landmarks.landmark[idx]
        # Convert normalized to pixel coords
        px, py = int(lm.x * w), int(lm.y * h)

        # Extract small 10x10 ROI around the cheek point
        # Safety check for image bounds
        if py < 5 or py > h-5 or px < 5 or px > w-5:
            continue

        roi = frame[py-5:py+5, px-5:px+5]

        # Calculate Redness: Mean(Red) / Mean(Green)
        # This helps normalize for overall brightness changes
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
    if len(history) > 60:  # 2 seconds
        history.pop(0)

    # Logic: Is current redness 10% higher than average of last 2 seconds?
    baseline = sum(history) / len(history)

    return avg_redness > (baseline * 1.10)


def calculate_blink_rate(upper_lid, lower_lid, current_rate):
    if abs(upper_lid.y - lower_lid.y) < 0.005:
        return current_rate + 1
    return current_rate


def detect_pupil_constriction(iris_landmarks):
    # Logic requires calculating the diameter of the dark pupil area vs iris
    return pupil_diameter < baseline_diameter * 0.8


def detect_teeth_sucking(face_landmarks):
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    # Logic: Lateral stretching + specific mouth shape
    return abs(left_corner.x - right_corner.x) > 0.08


def detect_eyebrow_flash(face_landmarks):
    # Measures quick lift of eyebrows (Landmark 105 vs 33) [cite: 47, 51]
    dist = abs(face_landmarks.landmark[105].y - face_landmarks.landmark[33].y)
    return dist > 0.04


def detect_teeth_sucking(face_landmarks):
    # Detects lateral lip tension (61, 291) + specific mouth shape [cite: 65, 67]
    width = abs(face_landmarks.landmark[61].x - face_landmarks.landmark[291].x)
    return width > 0.08


def detect_object_in_mouth(face_landmarks):
    # Detects if a hand/object (from pose) or lips (from face) pass the barrier [cite: 74, 77]
    # Here: simplified to detect lip-biting (lips passing teeth) [cite: 78]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    return abs(upper_lip.z - lower_lip.z) > 0.01


def detect_jaw_clenching(face_landmarks):
    # Monitors muscular tension in temple (21) and jaw (172) area [cite: 80, 81]
    dist = abs(face_landmarks.landmark[21].y - face_landmarks.landmark[172].y)
    return dist < 0.05


def detect_nostril_dilation(face_landmarks):
    # Measures horizontal distance between nostril wings (Landmarks 64 and 294)
    left_wing = face_landmarks.landmark[64]
    right_wing = face_landmarks.landmark[294]
    width = abs(left_wing.x - right_wing.x)
    return width > 0.035  # Threshold for 'flaring'


def detect_confirmation_glance(face_landmarks):
    # Detects a quick, cursory lateral shift of the pupils (Landmarks 473, 468)
    # compared to head orientation to verify if a story is "working"
    left_iris = face_landmarks.landmark[468]
    # Logic: Pupil position is significantly offset from center of eye socket
    return abs(left_iris.x - 0.5) > 0.02


def detect_yawn(face_landmarks):
    # Measures the vertical distance of the lower jaw (Landmark 17)
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    gap = abs(upper_lip.y - lower_lip.y)
    return gap > 0.15  # Distinctly larger than normal speech opening


def detect_happiness_genuine(face_landmarks):
    # Duchenne smile: checks for both lip corner pull (61, 291)
    # and eye crinkling (Orbicularis oculi tension near 33, 263)
    lip_width = abs(
        face_landmarks.landmark[61].x - face_landmarks.landmark[291].x)
    eye_squint = abs(
        face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
    return lip_width > 0.07 and eye_squint < 0.005


def detect_flushing(frame, face_landmarks):
    # Requires Raw Frame: Analyzes RGB/Lab color space in cheek regions
    # (Landmarks 205 for left cheek, 425 for right cheek)
    # This returns True if the 'a' channel (redness) spikes from baseline
    return check_redness_spike(frame, [205, 425])


def detect_head_back(face_landmarks):
    # Measures the pitch of the head relative to the camera/vertical plane
    nose = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    # Logic: Nose Y is significantly higher than neutral baseline relative to chin
    return nose.y < (chin.y - 0.15)


def detect_orbital_tension(face_landmarks):
    # Muscular tension around eyes (Orbicularis oculi) [cite: 115]
    # Measures eye opening height (Landmarks 159 vs 145)
    upper_lid = face_landmarks.landmark[159]
    lower_lid = face_landmarks.landmark[145]
    eye_opening = abs(upper_lid.y - lower_lid.y)
    # Small opening indicates tension [cite: 116]
    return 0.002 < eye_opening < 0.005


def detect_eyebrow_narrowing(face_landmarks):
    # Measures distance between inner eyebrows (Landmarks 55 and 285)
    left_inner = face_landmarks.landmark[55]
    right_inner = face_landmarks.landmark[285]
    return abs(left_inner.x - right_inner.x) < 0.02


def detect_vertical_head_shake(face_landmarks):
    # Monitors the pitch (Y-axis) of the nose (1) relative to
    # stationary shoulders to detect 'Yes' motion[cite: 147].
    return detect_oscillating_vertical_movement(face_landmarks.landmark[1])


def detect_head_support(face_landmarks, pose_landmarks):
    # Detects if the chin (152) or ear (7) rests on a hand (15, 16).
    # Hs1: Chin on hand. Hs2: Head tilted to hand near ear.
    chin = face_landmarks.landmark[152]
    left_hand = pose_landmarks.landmark[15]
    return abs(chin.y - left_hand.y) < 0.05 and abs(chin.x - left_hand.x) < 0.05


def detect_surprise_genuine(face_landmarks):
    # Genuine surprise (Doc #25): Eyebrows rise, exposing sclera (white part)
    # and the lower jaw drops[cite: 158, 159].
    eyebrow_dist = abs(
        face_landmarks.landmark[105].y - face_landmarks.landmark[33].y)
    jaw_gap = abs(
        face_landmarks.landmark[13].y - face_landmarks.landmark[14].y)
    return eyebrow_dist > 0.05 and jaw_gap > 0.08


def detect_pupil_dilation(face_landmarks):
    # Dilation (Doc #27): Response to pleasurable stimuli[cite: 174, 175].
    # Constriction (Doc #42): Response to aversion or disgust[cite: 176, 255].
    # Logic: Requires high-res iris tracking to measure dark center ratio.
    return detect_iris_center_ratio(face_landmarks, "expansion")


def detect_eye_squint(face_landmarks):
    """
    Detects narrowing of the eyelids (Squinting).
    Doc #28 [cite: 179]
    """
    lm = face_landmarks.landmark

    # Left Eye: Top (159) vs Bottom (145)
    left_open = abs(lm[159].y - lm[145].y)

    # Right Eye: Top (386) vs Bottom (374)
    right_open = abs(lm[386].y - lm[374].y)

    # Normalize by face height to account for camera zoom
    face_height = abs(lm[10].y - lm[152].y)
    if face_height == 0:
        return False

    avg_opening_ratio = ((left_open + right_open) / 2) / face_height

    # Threshold: Normal opening is ~0.05. Squint is tighter.
    # Blink is ~0.0. We want "Open but Small".
    return 0.005 < avg_opening_ratio < 0.03


def detect_disgust(face_landmarks):
    # Doc #31: Face 'crumples' toward the nose.
    # Logic: Nose (1), Brows (55, 285), and Lip (0) move toward center.
    return measure_facial_compression(face_landmarks)


def detect_fear(face_landmarks):
    # Doc #32: Eyes widen (sclera exposed), lips stretch horizontally.
    eye_sclera = abs(
        face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
    lip_stretch = abs(
        face_landmarks.landmark[61].x - face_landmarks.landmark[291].x)
    return eye_sclera > 0.05 and lip_stretch > 0.08


def detect_hushing(face_landmarks, pose_landmarks):
    # Subjects bringing hands to face or covering the mouth[cite: 399].
    # Logic: Proximity of wrist/fingers (15, 16) to mouth (13, 14).
    mouth = face_landmarks.landmark[13]
    hand = pose_landmarks.landmark[15]
    return abs(mouth.x - hand.x) < 0.05 and abs(mouth.y - hand.y) < 0.05


def detect_finger_to_nose(face_landmarks, pose_landmarks):
    # Variation of hushing that conceals the instinct to cover the mouth[cite: 473].
    # Logic: High-precision contact between index finger and nose tip (1).
    nose = face_landmarks.landmark[1]
    finger_tip = pose_landmarks.landmark[19]
    return abs(nose.x - finger_tip.x) < 0.02 and abs(nose.y - finger_tip.y) < 0.02


def detect_eyelid_rubbing(face_landmarks, pose_landmarks):
    # Mostly performed by men; indicates a need to end a train of thought[cite: 465].
    # Logic: Hand (15, 16) contact with ocular region (33, 263).
    eye = face_landmarks.landmark[33]
    hand = pose_landmarks.landmark[15]
    return abs(eye.x - hand.x) < 0.03 and abs(eye.y - hand.y) < 0.03
