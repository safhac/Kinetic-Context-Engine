from shared.sensor_interface import SensorInterface
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
# Fix import path to find the shared folder at root
import sys
import os
sys.path.append(os.getcwd())

# from body_worker.pose_signals import detect_body_gestures
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
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # 1. Check Environment Variable (Default to False for stability)
        enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"

        # 2. Select Delegate Dynamically
        if enable_gpu:
            print("🚀 Attempting to use GPU Delegate for Body Sensor...")
            selected_delegate = BaseOptions.Delegate.GPU
        else:
            print("💻 Using CPU Delegate for Body Sensor (Default).")
            selected_delegate = BaseOptions.Delegate.CPU

        # 3. Apply options
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='pose_landmarker.task',
                delegate=selected_delegate  # <--- Dynamic Selection
            ),
            running_mode=VisionRunningMode.VIDEO
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results_list = []

        # 1. Convert to MediaPipe Image (Required for new API)
        # Note: MediaPipe tasks expect RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Use the NEW Landmarker (which uses your CPU/GPU settings)
        # Timestamp must be in milliseconds (int)
        detection_result = self.landmarker.detect_for_video(
            mp_image, int(timestamp))

        # 3. Extract Landmarks
        # The new API returns a list of lists (one list per person detected)
        if detection_result.pose_landmarks:
            # We assume single person for now (index 0)
            landmarks = detection_result.pose_landmarks[0]

            # 4. Pass to your logic
            # IMPORTANT: 'landmarks' here is a list of NormalizedLandmark objects.
            # If 'detect_body_gestures' expects the old protobuf format,
            # you might need to adjust it slightly, but usually it's compatible
            # (both have x, y, z, visibility attributes).
            detected_signals = detect_body_gestures(landmarks)

            for signal in detected_signals:
                signal['timestamp'] = timestamp
                signal['source'] = 'mediapipe_body'
                results_list.append(signal)

        return results_list
