import math

# --- State Storage ---
_pose_history = {
    "throat_y": [],
    "left_ankle_pos": [],   # Stores tuples (x, y, z)
    "right_ankle_pos": [],
    "left_wrist_y": []      # For baton gestures
}

# --- Main Signal Aggregator ---


def get_active_pose_signals(pose_landmarks, object_centroids=None, other_property_bounds=None, audio_input=None):
    """
    Checks ALL implemented signals against the current frame.
    Args:
        pose_landmarks: List of NormalizedLandmark objects from MediaPipe.
        object_centroids: List of (x,y,z) tuples for external objects (optional).
        other_property_bounds: Dict {'x_min', ...} for property interaction (optional).
        audio_input: Audio rhythm data for sync detection (optional).
    """
    signals = []

    # --- CRITICAL FIX: The input IS the list of landmarks ---
    lm = pose_landmarks

    # --- Basic Posture ---
    if detect_head_downcast(lm):
        signals.append("head_down")

    shrug = detect_shoulder_shrug(lm)
    if shrug:
        signals.append(shrug)

    # --- Hand/Arm Gestures ---
    if detect_protecting_gesture(lm):
        signals.append("arms_crossed")
    if detect_steepling(lm):
        signals.append("steepling")
    if detect_elbow_closure(lm):
        signals.append("elbow_closure")
    if detect_ventilation(lm):
        signals.append("ventilation")
    if detect_security_check(lm):
        signals.append("security_check")

    # --- Leg/Foot Gestures ---
    if detect_foot_withdrawal(lm):
        signals.append("foot_withdrawal")
    if detect_binding_legs(lm):
        signals.append("binding_legs")
    if detect_inward_toe_pointing(lm):
        signals.append("inward_toe_pointing")

    # --- Rhythmic/Micro Expressions ---
    if detect_adams_apple_jump(lm):
        signals.append("adams_apple_jump")

    if detect_baton_gestures(lm):
        signals.append("baton_gesture")
        # Advanced: Check sync if audio is provided
        if audio_input and detect_asynchronous_baton(lm, audio_input):
            signals.append("async_baton_gesture")

    # --- Interaction with Environment ---
    if object_centroids and detect_object_barrier(lm, object_centroids):
        signals.append("object_barrier")

    if other_property_bounds and detect_property_interaction(lm, other_property_bounds):
        signals.append("interaction_others_property")

    # --- Status Indicators ---
    # Pelvic Tilt returns a specific state string
    tilt = detect_pelvic_tilt(lm)
    if tilt:
        signals.append(f"pelvic_tilt_{tilt}")

    # Generic Hand Raise (Wrist higher than Nose)
    if lm[15].y < lm[0].y:
        signals.append("hand_raise")

    return signals


# --- Helpers ---

def is_inside(point, bounds):
    """Checks if a point is inside a bounding box dict."""
    if not bounds:
        return False
    return (bounds['x_min'] <= point.x <= bounds['x_max']) and \
           (bounds['y_min'] <= point.y <= bounds['y_max'])


def _dist(p1, p2):
    """Euclidean distance ignoring Z depth"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def detect_sudden_velocity(landmark, history_key, threshold=0.05):
    """Detects high-velocity movement in 3D space."""
    history = _pose_history.get(history_key, [])
    current_pos = (landmark.x, landmark.y, landmark.z)

    history.append(current_pos)
    if len(history) > 2:
        history.pop(0)
    _pose_history[history_key] = history

    if len(history) < 2:
        return False

    prev, curr = history[0], history[1]
    dist = math.sqrt((curr[0]-prev[0])**2 +
                     (curr[1]-prev[1])**2 + (curr[2]-prev[2])**2)
    return dist > threshold


def detect_vertical_surge(landmarks, area="throat", threshold=0.015):
    """Detects a sudden upward movement (surge) in a specific body area."""
    current_y = 0.0
    history_key = ""

    if area == "throat":
        # Midpoint of shoulders approximates neck/throat base
        current_y = (landmarks[11].y + landmarks[12].y) / 2
        history_key = "throat_y"

    history = _pose_history.get(history_key, [])
    history.append(current_y)

    # Keep history short for immediate comparison
    if len(history) > 5:
        history.pop(0)
    _pose_history[history_key] = history

    if len(history) < 2:
        return False

    # Compare current frame vs average of previous frames to reduce jitter
    previous_avg = sum(history[:-1]) / len(history[:-1])
    diff = previous_avg - current_y  # Positive if moving UP (Y decreases)
    return diff > threshold


def detect_rhythmic_punctuation(landmark, history_key="left_wrist_y"):
    """Detects rapid up/down oscillation of wrist (Baton gestures)."""
    history = _pose_history.get(history_key, [])
    history.append(landmark.y)
    if len(history) > 20:
        history.pop(0)
    _pose_history[history_key] = history

    if len(history) < 20:
        return False

    # Count zero crossings in velocity
    velocities = [history[i] - history[i-1] for i in range(1, len(history))]
    flips = 0
    for i in range(1, len(velocities)):
        if (velocities[i] > 0 and velocities[i-1] < 0) or \
           (velocities[i] < 0 and velocities[i-1] > 0):
            if abs(velocities[i]) > 0.01:  # Threshold to ignore noise
                flips += 1

    return flips > 3  # At least 3 directional changes in ~0.6s


# --- Signal Implementations ---

def detect_head_downcast(lm):
    # Head lowered relative to shoulder line
    nose = lm[0]
    avg_shoulder_y = (lm[11].y + lm[12].y) / 2
    return nose.y > (avg_shoulder_y - 0.1)


def detect_shoulder_shrug(lm):
    # Measures rise of shoulders relative to nose
    nose_y = lm[0].y
    left_dist = abs(lm[11].y - nose_y)
    right_dist = abs(lm[12].y - nose_y)

    # Threshold: smaller distance means shoulders went UP
    threshold = 0.12
    is_left = left_dist < threshold
    is_right = right_dist < threshold

    if is_left and is_right:
        return "double_shrug"
    if is_left or is_right:
        return "single_shrug"
    return None


def detect_protecting_gesture(lm):
    # Arm crossing: Wrist crossing the midline of the body
    mid_x = (lm[11].x + lm[12].x) / 2
    # Check if either wrist has crossed the center X coordinate
    return (lm[15].x < mid_x) or (lm[16].x > mid_x)


def detect_steepling(lm):
    # Fingertips touching (Index fingers 19, 20)
    return _dist(lm[19], lm[20]) < 0.05


def detect_elbow_closure(lm):
    # Elbows drawn in toward body
    dist = _dist(lm[13], lm[14])
    # Threshold depends on body size, but < 0.3 is generally tight
    return dist < 0.3


def detect_ventilation(lm):
    # Pulling shirt collar: Hand near neck moving outward
    collar_area_y = (lm[11].y + lm[12].y) / 2
    l_hand = lm[15]
    # Hand near neck Y-level AND inside shoulder width
    return abs(l_hand.y - collar_area_y) < 0.1 and l_hand.x < lm[11].x


def detect_security_check(lm):
    # Checking pockets/hips
    # Check proximity of wrists (15, 16) to hips (23, 24)
    left_check = _dist(lm[15], lm[23]) < 0.15
    right_check = _dist(lm[16], lm[24]) < 0.15
    return left_check or right_check


def detect_foot_withdrawal(lm):
    # Sudden movement of ankles
    # ensure the function name is 'detect_sudden_velocity'
    left_move = detect_sudden_velocity(lm[27], "left_ankle_pos", 0.05)
    right_move = detect_sudden_velocity(lm[28], "right_ankle_pos", 0.05)
    return left_move or right_move


def detect_binding_legs(lm):
    # Ankles locked together
    return _dist(lm[27], lm[28]) < 0.1


def detect_inward_toe_pointing(lm):
    # Toes pointing inward
    # Vector: Heel to Foot Index
    l_toe_vec = lm[31].x - lm[29].x
    r_toe_vec = lm[32].x - lm[30].x
    # Left toe points Right (>0), Right toe points Left (<0)
    return l_toe_vec > 0 and r_toe_vec < 0


def detect_pelvic_tilt(lm):
    # Forward (confidence) vs Backward (retreat)
    mid_hip_z = (lm[23].z + lm[24].z) / 2
    mid_shoulder_z = (lm[11].z + lm[12].z) / 2
    return "forward" if mid_hip_z < mid_shoulder_z else "backward"


def detect_adams_apple_jump(lm):
    # Sudden rise of the throat area
    return detect_vertical_surge(lm, area="throat")


def detect_baton_gestures(lm):
    # Rhythmic hand beating
    return detect_rhythmic_punctuation(lm[15])


def detect_asynchronous_baton(lm, audio_input):
    # Placeholder logic: Compares baton rhythm with speech rhythm.
    # Returns True if NOT synced.
    #
    is_moving = detect_rhythmic_punctuation(lm[15])
    # Assume audio_input has property .is_speaking or .amplitude
    # This is a stub for the logic described in File 1
    if is_moving and audio_input and not audio_input.is_speaking:
        return True
    return False


def detect_object_barrier(lm, object_centroids):
    # Object placed between subject and camera
    torso_z = lm[1].z
    for obj in object_centroids:
        # Check if object Z is closer to camera than torso Z
        if obj.z < torso_z:
            # Simplified "in front" check: Object X is within shoulder width
            shoulders_min_x = min(lm[11].x, lm[12].x)
            shoulders_max_x = max(lm[11].x, lm[12].x)
            if shoulders_min_x < obj.x < shoulders_max_x:
                return True
    return False


def detect_property_interaction(lm, other_property_bounds):
    # Wrist entering someone else's space
    if is_inside(lm[15], other_property_bounds):
        return True
    return False
