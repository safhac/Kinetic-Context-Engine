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

        # 1. FIX PATH: Find the model file reliably
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pose_landmarker.task')

        if not os.path.exists(model_path):
            print(f"⚠️ WARNING: Model file not found at: {model_path}")
            print("Did you run 'wget' to download pose_landmarker.task?")

        # 2. Check Environment Variable (Default to False)
        enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"

        # 3. Select Delegate Dynamically
        if enable_gpu:
            print("🚀 Body Worker: Attempting to use GPU Delegate...")
            selected_delegate = BaseOptions.Delegate.GPU
        else:
            print("💻 Body Worker: Using CPU Delegate (Default).")
            selected_delegate = BaseOptions.Delegate.CPU

        # 4. Apply options
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,  # <--- Uses valid path
                delegate=selected_delegate    # <--- Uses dynamic hardware
            ),
            running_mode=VisionRunningMode.VIDEO
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results_list = []

        # MediaPipe Tasks API requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        detection_result = self.landmarker.detect_for_video(
            mp_image, int(timestamp))

        # Extract Results
        if detection_result.pose_landmarks:
            # Get the first person detected
            detected_signals_raw = get_active_pose_signals(landmarks)

            for signal_name in detected_signals_raw:
                # Convert the string name into a dictionary object
                signal_obj = {
                    "type": signal_name,  # e.g. "head_down"
                    "timestamp": timestamp,
                    "source": "mediapipe_body",
                    "confidence": 1.0  # Placeholder
                }
                results_list.append(signal_obj)

        return results_list
