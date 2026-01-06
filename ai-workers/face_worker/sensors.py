import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
from shared.sensor_interface import SensorInterface
from .face_signals import detect_face_gestures


class MediaPipeFaceSensor(SensorInterface):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.face_mesh.process(rgb_frame)

        if processed.multi_face_landmarks:
            for face_landmarks in processed.multi_face_landmarks:
                # NEW WAY (Using your new file):
                # Pass the raw landmarks to the new library
                detected_signals = detect_face_gestures(face_landmarks)

                # Add timestamp and source
                for signal in detected_signals:
                    signal['timestamp'] = timestamp
                    signal['source'] = 'mediapipe_face'
                    results.append(signal)

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
