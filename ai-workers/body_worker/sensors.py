from shared.sensor_interface import SensorInterface
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
# Fix import path to find the shared folder at root
import sys
import os
sys.path.append(os.getcwd())

# --- FIX: Direct import + Correct function name ---
try:
    from pose_signals import get_active_pose_signals
except ImportError:
    from .pose_signals import get_active_pose_signals


class SensorInterface:
    def process_frame(self, frame, timestamp):
        raise NotImplementedError


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
            # Delegate logic to the new file
            detected_signals = detect_body_gestures(processed.pose_landmarks)

            for signal in detected_signals:
                signal['timestamp'] = timestamp
                signal['source'] = 'mediapipe_body'
                results.append(signal)

        return results
