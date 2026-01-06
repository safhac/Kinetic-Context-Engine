# Mapping library-specific detections to your Gestures Doc signals
SIGNAL_ADAPTER = {
    "mediapipe": {
        "eyebrow_raise": "signal_eyebrow_flash",
        "blink_count": "signal_blink_rate",
        "lip_press": "signal_lip_compression",
        "chin_protrusion": "chin_thrust",
        "elbow_inward_dip": "signal_elbow_closure",
        "hand_on_knee_grip": "signal_knee_clasp",
        "palm_eye_rub": "signal_eyelid_rubbing",
        "general_face_contact": "signal_facial_touching",
        "hand_to_pocket_check": "signal_security_check",
        "object_in_chest_zone": "signal_object_barrier",
        "rapid_object_velocity": "signal_carelessness",
        "eye_blink": "signal_blink_rate",
        "eye_squint": "signal_eye_squint",
        "eye_wide": "signal_surprise",
        "nose_flare": "signal_nostril_dilation",

        # Body & Hands
        "wrists_crossed": "signal_arm_cross",
        "shoulders_up": "signal_turtling",
        "hand_to_collar": "signal_ventilation",
        "fingertips_touching": "signal_steepling",
        "hand_to_nose": "signal_finger_to_nose",
        "hand_to_eye": "signal_eyelid_rubbing",
        "hand_to_mouth": "signal_hushing",
        "ankles_locked": "signal_locked_ankles",
        "knee_grab": "signal_knee_clasp",
        "torso_lean_forward": "signal_posture_tilt_toward",
        "torso_lean_backward": "signal_posture_tilt_away",
        # Brows
        "brow_down_left": "signal_eyebrow_narrowing",
        "brow_down_right": "signal_eyebrow_narrowing",
        "brow_inner_up": "signal_eyebrow_flash",

        # Eyes
        "eye_blink_left": "signal_blink_rate",
        "eye_blink_right": "signal_blink_rate",
        "eye_squint_left": "signal_eye_squint",
        "eye_wide_left": "signal_surprise",

        # Head & Face
        "head_pitch_up": "signal_head_back",
        "head_down": "signal_head_downcast",
        "ear_slope": "signal_head_tilt",
        "nose_vertical_oscillation": "signal_head_shake_vertical",
        "head_horizontal_oscillation": "signal_head_shake_no",
        "brow_constantly_up": "signal_constantly_raised_eyebrows",
        "orbital_muscle_contract": "signal_orbital_tension",
        "pupil_shrink": "signal_pupil_constriction",
        "mouth_close": "signal_lip_compression",
        "mouth_pucker": "signal_hushing",
        "mouth_down_corners": "signal_sadness",
        "nose_bridge_wrinkle": "signal_disgust",
        "jaw_forward": "signal_chin_thrust",
        "jaw_open": "signal_yawn",

        # Hands & Arms
        "wrists_crossed_torso": "signal_arm_cross",
        "thumb_up_cross": "signal_arm_cross_thumbs_up",
        "clenched_fist_cross": "signal_arm_cross_fists",
        "rhythmic_wrist_flick": "signal_baton_gestures",
        "chin_hand_contact": "signal_head_support",
        "hand_to_mouth_proximity": "signal_hushing",
        "index_nose_touch": "signal_finger_to_nose",
        "hand_to_throat_clasp": "signal_throat_clasping",
        "hand_to_chest_touch": "signal_chest_touching",
        "hand_to_back_neck": "signal_hands_on_back_neck",
        "hand_wrist_to_forehead": "signal_wrist_to_forehead",
        "fingertip_pyramid": "signal_steepling",
        "palm_facing_down": "signal_palms_down",
        "hand_grabbing_wrist": "signal_wrist_touch_grabbing",
        "hand_to_jewelry_fidget": "signal_jewelry_play",

        # Lower Body & Torso
        "shoulders_raised": "signal_turtling",
        "shoulder_asymmetric_shrug": "single_shrug",
        "torso_lean_backward": "signal_posture_tilt_away",
        "pelvis_forward_tilt": "signal_pelvic_tilt_forward",
        "ankle_contact_static": "signal_binding_legs",
        "ankle_on_opposite_knee": "signal_figure_four_cross",
        "leg_over_knee_and_wrap": "signal_double_leg_cross",
        "legs_spread_open": "signal_groin_exposure",
        "traditional_leg_cross": "signal_leg_crossing",
        "shoe_toe_lift": "signal_toes_up",
        # To be added to mediapipe
        "hand_shielding_groin": "signal_genital_protecting",
        "object_over_genitals": "signal_groin_shield",

        # To be added to mmpose
        "keypoint_groin_protection": "signal_genital_protecting"
    },
    "openface": {
        "AU01": "eyebrow_flash",
        "AU02": "eyebrow_flash",
        "AU24": "lip_compression",
        "AU17": "chin_thrust",
        "AU01_02_intensity": "signal_eyebrow_flash",
        "AU45_r": "signal_blink_rate",
        "AU24_c": "signal_lip_compression",
        "AU04": "signal_eyebrow_narrowing",  # Brow Lowerer
        "AU07": "signal_orbital_tension",  # Lid Tightener
        # Often correlates with brow-rubbing
        "AU04_intensity_high": "signal_facial_touching",
        "AU01_c": "signal_eyebrow_flash",
        "AU04_c": "signal_eyebrow_narrowing",
        "AU17_c": "signal_chin_thrust",
        "AU45_c": "signal_blink_rate",
        "AU26_c": "signal_surprise",
        "AU01": "signal_eyebrow_flash",
        "AU02": "signal_eyebrow_flash",
        "AU04": "signal_eyebrow_narrowing",
        "AU06": "signal_happiness",
        "AU07": "signal_orbital_tension",
        "AU09": "signal_disgust",
        "AU10": "signal_disgust",
        "AU12": "signal_happiness",
        "AU15": "signal_sadness",
        "AU17": "signal_chin_thrust",
        "AU20": "signal_fear",
        "AU24": "signal_lip_compression",
        "AU26": "signal_surprise",
        "AU45": "signal_blink_rate",
        "AU04_intensity_high": "signal_facial_touching",
        "AU02_c": "signal_eyebrow_flash",
        "AU07_c": "signal_orbital_tension",
        "AU12_c": "signal_happiness",
        "AU04_r": "signal_facial_touching"
    },
    "mmpose": {
        "keypoint_0_low": "head_downcast",
        "shoulders_up": "turtling",
        "head_angle_y_negative": "signal_head_downcast",
        "shoulder_y_high": "signal_turtling",
        "shoulder_y_diff_low": "signal_turtling",
        "neck_vertical_movement": "signal_adams_apple_jump",
        "neck_surge": "signal_adams_apple_jump",
        "wrist_speed_spike": "signal_baton_gestures",
        "wrist_oscillation": "signal_baton_gestures",
        "keypoint_ankles_locked": "signal_locked_ankles",
        "elbow_torso_proximity": "signal_elbow_closure"
    },
    "OpenPose": {
        "neck_shorten": "turtling",
        "arm_intersect": "arm_cross"
    },
    "AudioAnalyzer": {
        "pitch_deviation_pos": "signal_pitch_increase",
        "words_per_minute_high": "signal_vocal_speed"  # Doc #111
    },
    "NLP_Engine": {
        "euphemism_detected": "signal_distancing_words",
        "zero_pronouns": "signal_no_pronouns",
        "exclusion_phrases": "signal_exclusion_words"
    }
}
