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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            # 1. FIX: Set to True to force processing frames independently (CPU-safe)
            # 'False' treats it as a video stream and tries to open an OpenGL context for tracking.
            static_image_mode=True,

            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = []

        # 1. Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # In the legacy API, .process() is used for both static and video modes
        processed = self.face_mesh.process(rgb_frame)

        if processed.multi_face_landmarks:
            for face_landmarks in processed.multi_face_landmarks:

                # 2. CALL SIGNAL DETECTION WITH FRAME
                active_signals = get_active_face_signals(
                    face_landmarks=face_landmarks,
                    pose_landmarks=None,
                    frame=frame
                )

                # 3. Format Output
                for signal_name in active_signals:
                    results.append({
                        "signal": signal_name,
                        "intensity": 1.0,
                        "timestamp": timestamp,
                        "source": "mediapipe_face_mesh"
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
