import sys
import os
import math

# Add project root to path so we can import 'shared'
sys.path.append(os.getcwd())

try:
    from shared.adapter import SIGNAL_ADAPTER
except ImportError:
    # Fallback if running from a different context
    print("⚠️ Warning: Could not import shared.adapter, using empty map.")
    SIGNAL_ADAPTER = {"mediapipe": {}}


class BToEAdapter:
    """
    detects raw geometric states (like 'wrists_crossed') 
    and maps them to Canonical Signals using the Shared Adapter.
    """

    def __init__(self):
        self.mp_map = SIGNAL_ADAPTER.get("mediapipe", {})

        # Reverse lookup for meanings (Optional, if we want text descriptions in VTT)
        # For now, we return the Signal ID (e.g., 'signal_arm_cross')

    def _get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def analyze_frame(self, landmarks):
        """
        Input: MediaPipe Landmarks
        Output: List of Canonical Signal IDs (e.g. ['signal_arm_cross'])
        """
        detected_raw_signals = []

        if not landmarks:
            return []

        # -- 1. EXTRACT KEYPOINTS --
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        l_elbow = landmarks[13]
        r_elbow = landmarks[14]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]

        # Normalizer (Shoulder Width)
        shoulder_width = self._get_distance(l_shoulder, r_shoulder)
        if shoulder_width == 0:
            return []

        # -- 2. GEOMETRY CHECKS --

        # A. Wrists Crossed (Arm Cross)
        # Check if wrists are close to each other OR close to opposite elbows
        wrist_dist = self._get_distance(l_wrist, r_wrist)
        l_wrist_r_elbow = self._get_distance(l_wrist, r_elbow)

        # Threshold: 30% of shoulder width
        if wrist_dist < (0.3 * shoulder_width) or l_wrist_r_elbow < (0.4 * shoulder_width):
            detected_raw_signals.append("wrists_crossed")

        # B. Hands on Face (Hushing / Thinking)
        # Check if wrists are above the shoulders (y is smaller when higher)
        nose = landmarks[0]
        if l_wrist.y < l_shoulder.y and self._get_distance(l_wrist, nose) < (0.3 * shoulder_width):
            detected_raw_signals.append("hand_to_mouth_proximity")

        # -- 3. MAPPING --
        # Convert "wrists_crossed" -> "signal_arm_cross" using your shared adapter
        canonical_signals = []
        for raw in detected_raw_signals:
            if raw in self.mp_map:
                canonical_signals.append(self.mp_map[raw])
            else:
                canonical_signals.append(raw)  # Fallback

        return canonical_signals
