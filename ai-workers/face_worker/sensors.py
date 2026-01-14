import os
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
from shared.sensor_interface import SensorInterface
try:
    from face_signals import get_active_face_signals
except ImportError:
    # Fallback if running from a different directory context
    from .face_signals import get_active_face_signals


class SensorInterface:
    def process_frame(self, frame, timestamp):
        raise NotImplementedErro


class MediaPipeFaceSensor(SensorInterface):
    def __init__(self):
        # Using the New Task API (v2) to match Body Worker
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'face_landmarker.task')

        if not os.path.exists(model_path):
            print(f"❌ CRITICAL: Face model not found at {model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,  # Prevents GL context crashes
            output_face_blendshapes=True,
            num_faces=1
        )
        # This makes 'self.landmarker' available for main.py
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """
        Legacy support for the process_frame interface.
        Note: main.py now calls self.landmarker.detect() directly for VTT.
        """
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            for landmarks in detection_result.face_landmarks:
                # Map detection using your face_signals logic
                active_signals = get_active_face_signals(
                    face_landmarks=landmarks,
                    pose_landmarks=None,
                    frame=frame
                )

                for signal_name in active_signals:
                    results.append({
                        "signal": signal_name,
                        "intensity": 1.0,
                        "timestamp": timestamp,
                        "source": "mediapipe_face_task"
                    })

        return results


class OpenFaceSensor(SensorInterface):
    """
    Wrapper for OpenFace Action Unit (AU) detection.
    Note: Requires OpenFace binary or library installed in container.
    """

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = []
        # Placeholder: In prod, this would call the OpenFace binary or C++ wrapper
        # simulating detection of Lip Compression (AU23 + AU24)

        # mock_detection_logic()
        # if au23_intensity > 0.7 and au24_intensity > 0.7:
        #     results.append({
        #         "signal": "lip_compression",
        #         "intensity": 0.85,
        #         "timestamp": timestamp,
        #         "source": "openface_au"
        #     })
        return results
