from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class SensorInterface(ABC):
    """
    Abstract Base Class for all KCE Sensation Workers.
    Enforces uniform signal output formats.
    """

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """
        Input: Raw numpy image array (cv2 format).
        Output: List of atomic signals.
        Example: [{'signal': 'eyebrow_raise', 'intensity': 0.8, 'source': 'mediapipe'}]
        """
        pass