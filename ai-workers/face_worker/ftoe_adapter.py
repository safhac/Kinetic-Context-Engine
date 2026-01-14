from face_worker.face_signals import get_active_face_signals
import sys
import os

# Add project root to path so we can import 'shared'
sys.path.append(os.getcwd())

try:
    from shared.adapter import SIGNAL_ADAPTER
except ImportError:
    print("⚠️ Warning: Could not import shared.adapter, using empty map.")
    SIGNAL_ADAPTER = {"mediapipe": {}, "openface": {}}

# IMPORT THE GEOMETRY ENGINE


class FToEAdapter:
    """
    The Brain (Face):
    1. Receives Raw Landmarks
    2. Asks face_signals.py: "What is the face doing?" (e.g., 'eyebrow_flash')
    3. Asks shared/adapter.py: "What does that mean in the Book?" (e.g., 'signal_eyebrow_flash')
    """

    def __init__(self):
        # We check both maps since keys might vary
        self.mp_map = SIGNAL_ADAPTER.get("mediapipe", {})
        self.of_map = SIGNAL_ADAPTER.get("openface", {})

    def analyze_frame(self, landmarks, frame=None):
        """
        Input: MediaPipe Landmarks, optional BGR Frame (for redness)
        Output: List of Canonical Signal IDs
        """
        if not landmarks:
            return []

        # 1. GET RAW GEOMETRY SIGNALS
        # We pass None for pose_landmarks (until we unify sensors)
        raw_detected = get_active_face_signals(
            landmarks, pose_landmarks=None, frame=frame)

        # 2. MAP TO BOOK CODES
        canonical_signals = []

        for raw_id in raw_detected:
            # Try Mapping
            if raw_id in self.mp_map:
                canonical_signals.append(self.mp_map[raw_id])
            elif raw_id in self.of_map:
                canonical_signals.append(self.of_map[raw_id])
            else:
                # Pass through raw if no mapping exists yet
                canonical_signals.append(raw_id)

        return canonical_signals
