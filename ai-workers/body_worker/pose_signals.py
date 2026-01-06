import math

# A dictionary to hold state between frames.
# In a real distributed system, this would be Redis, but for a worker process,
# a global or class-level dict works perfectly fine for sequential frames.
_pose_history = {
    "throat_y": [],
    "left_ankle_y": [],
    "right_ankle_y": []
}


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
    # Doc #103: Checking safety/location of personal items (wallet, phone, purse).
    # Logic: Hand (15, 16) moves to pockets or patting motion on clothing.
    return detect_patting_motion(pose_landmarks.landmark[15], "pockets")


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
