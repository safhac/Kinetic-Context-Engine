from body_worker.pose_signals import get_active_pose_signals
import sys
import os
import math

# Add project root to path so we can import 'shared'
sys.path.append(os.getcwd())

try:
    from shared.adapter import SIGNAL_ADAPTER
except ImportError:
    print("⚠️ Warning: Could not import shared.adapter, using empty map.")
    SIGNAL_ADAPTER = {"mediapipe": {}}

# IMPORT THE GEOMETRY ENGINE


class BToEAdapter:
    """
    The Brain:
    1. Receives Raw Landmarks
    2. Asks pose_signals.py: "What is the body doing physically?" (e.g., 'steepling')
    3. Asks shared/adapter.py: "What does that mean in the Book?" (e.g., 'signal_steepling')
    """

    def __init__(self):
        # Load the mapping for MediaPipe keys
        self.mp_map = SIGNAL_ADAPTER.get("mediapipe", {})

    def analyze_frame(self, landmarks):
        """
        Input: MediaPipe Landmarks
        Output: List of Canonical Signal IDs (e.g. ['signal_steepling', 'signal_arm_cross'])
        """
        if not landmarks:
            return []

        # 1. GET RAW GEOMETRY SIGNALS
        # We pass None for objects/audio for now, can hook those up later
        raw_detected = get_active_pose_signals(landmarks)

        # 2. MAP TO BOOK CODES
        canonical_signals = []

        for raw_id in raw_detected:
            # Try to find the raw_id in our Adapter Map
            # e.g., "arms_crossed" -> "signal_arm_cross"
            if raw_id in self.mp_map:
                canonical_signals.append(self.mp_map[raw_id])
            else:
                # If we don't have a map for it yet, pass it through raw
                # so we can see it in the VTT and fix the map later.
                canonical_signals.append(raw_id)

        return canonical_signals
