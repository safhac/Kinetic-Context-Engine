from audio_worker.verbal_signals import analyze_segment
import sys
import os
import parselmouth

sys.path.append(os.getcwd())

try:
    from shared.adapter import SIGNAL_ADAPTER
except ImportError:
    SIGNAL_ADAPTER = {"AudioAnalyzer": {}, "NLP_Engine": {}}

# IMPORT NEW ENGINE


class VToEAdapter:
    def __init__(self):
        self.audio_map = SIGNAL_ADAPTER.get("AudioAnalyzer", {})
        self.nlp_map = SIGNAL_ADAPTER.get("NLP_Engine", {})

    def analyze(self, sound_object, text=None, baseline_pitch=None):
        """
        Orchestrates the analysis and mapping.
        """
        raw_signals = analyze_segment(sound_object, text, baseline_pitch)

        canonical = []
        for raw in raw_signals:
            # Check both maps
            if raw in self.audio_map:
                canonical.append(self.audio_map[raw])
            elif raw in self.nlp_map:
                canonical.append(self.nlp_map[raw])
            else:
                canonical.append(raw)

        return canonical
