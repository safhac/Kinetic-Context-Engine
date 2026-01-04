import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any
from shared.sensor_interface import SensorInterface

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
                # Logic: Calculate Eyebrow Raise (Distance between eye top and eyebrow bottom)
                # Landmarks: 159 (Left Eye Top), 66 (Left Eyebrow Inner)
                left_eye_top = face_landmarks.landmark[159].y
                left_brow_inner = face_landmarks.landmark[66].y
                
                # Simple heuristic for normalized distance
                distance = abs(left_eye_top - left_brow_inner)
                
                # Threshold for "Raise" (Tune based on empirical data)
                if distance > 0.05: 
                    results.append({
                        "signal": "eyebrow_raise",
                        "intensity": min(distance * 10, 1.0), # Normalize 0-1
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