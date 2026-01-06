import math

# A dictionary to hold state between frames.
# In a real distributed system, this would be Redis, but for a worker process,
# a global or class-level dict works perfectly fine for sequential frames.
_pose_history = {
    "throat_y": [],
    "left_ankle_y": [],
    "right_ankle_y": [],
    "left_ankle_pos": [],  # Stores tuples (x, y, z)
    "right_ankle_pos": []
}


def detect_sudden_velocity(landmark, history_key, threshold=0.05):
    """
    Detects high-velocity movement in any direction.
    Args:
        landmark: The MediaPipe landmark object.
        history_key: String key to store history (e.g., 'left_ankle_pos').
        threshold: Distance moved per frame (0.05 is ~5% of screen width).
    """
    # 1. Get History
    history = _pose_history.get(history_key, [])

    # 2. Store current position
    current_pos = (landmark.x, landmark.y, landmark.z)
    history.append(current_pos)

    # Keep strictly last 2 frames for instantaneous velocity
    if len(history) > 2:
        history.pop(0)
    _pose_history[history_key] = history

    # 3. Calculate Velocity
    if len(history) < 2:
        return False

    prev = history[0]
    curr = history[1]

    # Euclidean distance formula (3D)
    dist = math.sqrt(
        (curr[0] - prev[0])**2 +
        (curr[1] - prev[1])**2 +
        (curr[2] - prev[2])**2
    )

    return dist > threshold


def detect_vertical_surge(pose_landmarks, area="throat", threshold=0.015):
    """
    Detects a sudden upward movement (surge) in a specific body area.
    Returns True if a surge occurred compared to recent history.
    """
    landmarks = pose_landmarks.landmark
    current_y = 0.0
    history_key = ""

    # 1. Determine the target Y coordinate based on 'area'
    if area == "throat":
        # Approximate Adam's Apple: Midpoint between shoulders (11, 12)
        # We assume the neck moves with the shoulders/head.
        # Ideally, we'd use face landmark 152 (chin) + offset, but we are in Body Worker.
        current_y = (landmarks[11].y + landmarks[12].y) / 2
        history_key = "throat_y"
    elif area == "ankle_left":
        current_y = landmarks[27].y
        history_key = "left_ankle_y"

    # 2. Retrieve History
    history = _pose_history.get(history_key, [])

    # 3. Update History (Keep last 5 frames for smoothing or immediate comparison)
    history.append(current_y)
    if len(history) > 5:
        history.pop(0)
    _pose_history[history_key] = history

    # 4. Check for Surge (Logic: Previous - Current > Threshold)
    # Note: In MediaPipe, Y=0 is TOP. So "Upward" movement means Y decreases.
    # Therefore: (Old_Y - New_Y) should be POSITIVE.

    if len(history) < 2:
        return False

    # Compare current frame vs the average of the last 3 frames (to reduce jitter)
    previous_avg = sum(history[:-1]) / len(history[:-1])

    diff = previous_avg - current_y  # Positive if moving UP

    return diff > threshold


def detect_adams_apple_jump(pose_landmarks):
    # Detects sudden rise of the throat area
    # We use a threshold of 0.015 (approx 1.5% of screen height)
    if detect_vertical_surge(pose_landmarks, area="throat", threshold=0.015):
        return True
    return False


def detect_foot_withdrawal(pose_landmarks):
    # Doc #91: Sudden withdrawal of feet under a chair.
    # Logic: High-velocity movement of ankles.
    # We check both ankles.
    left_move = detect_sudden_velocity(
        pose_landmarks.landmark[27], "left_ankle_pos", threshold=0.05
    )
    right_move = detect_sudden_velocity(
        pose_landmarks.landmark[28], "right_ankle_pos", threshold=0.05
    )

    return left_move or right_move


def detect_patting_motion(hand_landmark, hip_landmark):
    """
    Detects if a hand is hovering near a hip and moving slightly.
    """
    # 1. Proximity Check (Is hand near pocket?)
    # Distance between wrist and hip
    dist = math.sqrt(
        (hand_landmark.x - hip_landmark.x)**2 +
        (hand_landmark.y - hip_landmark.y)**2
    )

    # If hand is not near hip (0.15 radius), ignore
    if dist > 0.15:
        return False

    # 2. Movement Check
    # We can reuse 'detect_sudden_velocity' with a LOWER threshold
    # to detect the "jitter" of patting, or just return True for "Hand on Hip"
    # For MVP, let's detect "Hand touching Hip/Pocket area"
    return True


def detect_head_downcast(pose_landmarks):
    # Head lowered relative to shoulder line [cite: 54, 68]
    nose = pose_landmarks.landmark[0]
    avg_shoulder_y = (
        pose_landmarks.landmark[11].y + pose_landmarks.landmark[12].y) / 2
    return nose.y > (avg_shoulder_y - 0.05)


def detect_ventilation(pose_landmarks):
    # Logic: Hand (15 or 16) moves to neck/collar (11, 12)
    # followed by outward movement to 'pull' clothing [cite: 125]
    left_hand = pose_landmarks.landmark[15]
    collar_area = (
        pose_landmarks.landmark[11].y + pose_landmarks.landmark[12].y) / 2
    return abs(left_hand.y - collar_area) < 0.05


def detect_adams_apple_jump(pose_landmarks):
    # Requires high-resolution tracking of the throat area (Landmark 0 in some models)
    # Associated with the reticular activating system and stress [cite: 131, 132]
    return detect_vertical_surge(pose_landmarks, area="throat")


def detect_baton_gestures(pose_landmarks):
    # Detects hand motions that accentuate syllabic/emotional punctuation.
    # Measures the frequency and rhythm of wrist movement (15, 16).
    return detect_rhythmic_punctuation(pose_landmarks.landmark[15])


def detect_asynchronous_baton(pose_landmarks, audio_input):
    # Logic: Compares baton rhythm with speech rhythm.
    # A gap indicates the emotion is likely not genuine[cite: 141, 143].
    return not is_rhythm_synced(pose_landmarks.landmark[15], audio_input)


def detect_shoulder_shrug(pose_landmarks):
    # Measures sudden vertical rise of shoulders (11, 12) relative to the neck.
    # Can be Double (Doc #29) or Single (Doc #38).
    left_y = pose_landmarks.landmark[11].y
    right_y = pose_landmarks.landmark[12].y
    # Logic: Significant upward shift from baseline
    is_left_up = left_y < (baseline_y - 0.05)
    is_right_up = right_y < (baseline_y - 0.05)

    if is_left_up and is_right_up:
        return "double_shrug"
    if is_left_up or is_right_up:
        return "single_shrug"
    return None


def detect_protecting_gesture(pose_landmarks):
    # Limbs crossing over the body to cover vital areas (Doc #35).
    # Check if wrists (15, 16) cross the torso mid-line.
    return is_crossing_torso(pose_landmarks.landmark[15], pose_landmarks.landmark[16])


def detect_elbow_closure(pose_landmarks):
    # Measures the inward drawing of the elbows while seated (Doc #37).
    left_elbow = pose_landmarks.landmark[13]
    right_elbow = pose_landmarks.landmark[14]
    # Logic: Elbows move closer together toward the mid-line.
    return abs(left_elbow.x - right_elbow.x) < 0.2


def detect_steepling(pose_landmarks):
    # Palms face each other with fingertips touching (Doc #47).
    l_fingers = [pose_landmarks.landmark[i] for i in [17, 19, 21]]
    r_fingers = [pose_landmarks.landmark[i] for i in [18, 20, 22]]
    # Logic: Proximity between corresponding left/right fingertips.
    return is_fingertip_contact(l_fingers, r_fingers)


def detect_binding_legs(pose_landmarks):
    # Legs come together with no relaxation, often feet touching (Doc #83).
    left_ankle = pose_landmarks.landmark[27]
    right_ankle = pose_landmarks.landmark[28]
    return abs(left_ankle.x - right_ankle.x) < 0.05


def detect_object_barrier(pose_landmarks, object_centroids):
    # Doc #100: Putting objects between subject and interviewer.
    # Logic: An object (e.g., cup, book) is detected in the 'dead zone'
    # between the subject's torso and the camera.
    torso_z = pose_landmarks.landmark[1].z
    for obj in object_centroids:
        if obj.z < torso_z and is_in_front_of_chest(obj, pose_landmarks):
            return True
    return False


def detect_property_interaction(pose_landmarks, other_property_bounds):
    # Doc #93: Physical interaction with property belonging to others.
    # Logic: Wrist (15, 16) enters the bounding box of someone else's property.
    left_hand = pose_landmarks.landmark[15]
    if is_inside(left_hand, other_property_bounds):
        return "signal_interaction_others_property"
    return None


def detect_security_check(pose_landmarks):
    # Doc #103: Checking safety of personal items (pockets).
    # Check Left Hand vs Left Hip (23) AND Right Hand vs Right Hip (24)
    lm = pose_landmarks.landmark

    left_check = detect_patting_motion(lm[15], lm[23])
    right_check = detect_patting_motion(lm[16], lm[24])

    return left_check or right_check


def detect_pelvic_tilt(pose_landmarks):
    # Doc #59: Forward tilt (confidence) vs backward tilt (lack of confidence).
    # Logic: Angle between hip (23, 24) and shoulder (11, 12) on Z-axis.
    mid_hip_z = (pose_landmarks.landmark[23].z +
                 pose_landmarks.landmark[24].z) / 2
    mid_shoulder_z = (
        pose_landmarks.landmark[11].z + pose_landmarks.landmark[12].z) / 2
    return "forward" if mid_hip_z < mid_shoulder_z else "backward"


def detect_inward_toe_pointing(pose_landmarks):
    # Doc #88: Toes point inward (pigeon-toed).
    # Logic: Orientation of the vector between heel (29, 30) and index toe (31, 32).
    left_toe_vec = pose_landmarks.landmark[31].x - \
        pose_landmarks.landmark[29].x
    right_toe_vec = pose_landmarks.landmark[32].x - \
        pose_landmarks.landmark[30].x
    return left_toe_vec > 0 and right_toe_vec < 0


def detect_foot_withdrawal(pose_landmarks):
    # Doc #91: Sudden withdrawal of feet under a chair.
    # Logic: High-velocity negative Z/Y shift of ankles (27, 28) while seated.
    return detect_sudden_velocity(pose_landmarks.landmark[27], direction="back")
