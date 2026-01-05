import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
# Fix import path to find the shared folder at root
import sys
import os
sys.path.append(os.getcwd())
from shared.sensor_interface import SensorInterface

class MediaPipeBodySensor(SensorInterface):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = []
        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.pose.process(rgb_frame)

        if processed.pose_landmarks:
            landmarks = processed.pose_landmarks.landmark
            
            # --- LOGIC: HAND RAISE DETECTION ---
            # Landmarks: 0=Nose, 15=Left Wrist, 16=Right Wrist
            # Note: In MediaPipe, Y coordinates are normalized (0 = Top, 1 = Bottom).
            # So a "Higher" hand has a LOWER Y value.
            
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
            left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y

            # Threshold: Wrist must be significantly above nose (0.1 buffer)
            if left_wrist_y < (nose_y - 0.1) or right_wrist_y < (nose_y - 0.1):
                results.append({
                    "signal": "hand_raise",
                    "intensity": 1.0, 
                    "timestamp": timestamp,
                    "source": "mediapipe_body_pose"
                })
                
        return results