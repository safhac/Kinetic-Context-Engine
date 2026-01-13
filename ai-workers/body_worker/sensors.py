import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any

# Ensure we can find shared modules if needed
sys.path.append(os.getcwd())

# --- IMPORT FIX ---
try:
    from pose_signals import get_active_pose_signals
except ImportError:
    from .pose_signals import get_active_pose_signals


class SensorInterface:
    def process_frame(self, frame, timestamp):
        raise NotImplementedError


class MediaPipeBodySensor(SensorInterface):
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pose_landmarker.task')

        # ... (keep your existing GPU/CPU logic if you want, but strictly:) ...

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            # 1. CHANGE: Switch to IMAGE mode to avoid GL context crash
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results_list = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. CHANGE: Use .detect() instead of .detect_for_video()
        # Note: .detect() does NOT accept a timestamp
        detection_result = self.landmarker.detect(mp_image)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]

            # (Keep your existing signal logic)
            try:
                detected_signals_raw = get_active_pose_signals(landmarks)
            except NameError:
                # Fallback if function isn't imported correctly
                detected_signals_raw = []

            for signal_name in detected_signals_raw:
                signal_obj = {
                    "type": signal_name,
                    "timestamp": timestamp,  # Manually pass the timestamp here
                    "source": "mediapipe_body",
                    "confidence": 1.0
                }
                results_list.append(signal_obj)

        return results_list
