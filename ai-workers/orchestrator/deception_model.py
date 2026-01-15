import json
import os
from shared.schemas import GestureSignal


class DeceptionModel:
    def __init__(self):
        # Weights: How much does each signal contribute to the Deception Score?
        self.signal_weights = {
            # --- FACE (Visual) ---
            "chin_thrust": 0.6,
            "lips_compressed": 0.5,
            "eyebrow_flash": 0.3,
            "eye_squint": 0.4,
            "flushing": 0.9,
            "nodding_yes": 0.1,
            "head_down": 0.6,
            "confirmation_glance": 0.7,

            # --- BODY (Pose) ---
            "single_shrug": 0.5,
            "double_shrug": 0.4,
            "arms_crossed": 0.2,
            "hand_raise": 0.1,
            "steepling": -0.3,
            "hand_on_face": 0.6,
            "hushing": 0.7,
            "security_check": 0.5,
            "foot_withdrawal": 0.6,

            # --- AUDIO (Verbal & Paralinguistic) ---
            "psychological_distancing": 0.8,
            "pronoun_absence": 0.5,
            "vocal_pitch_rise": 0.7,
            "non_contracting_statement": 0.4,
            "over_apologizing": 0.5
        }

        self.score = 0.0
        self.decay_rate = 0.05

    def analyze(self, signal: GestureSignal) -> float:
        """The Bridge: Adapts the new Orchestrator call to your original logic."""
        self.decay()
        key = signal.text.lower().replace(" ", "_")
        new_score = self.update_score(key, intensity=signal.confidence)
        return new_score

    def update_score(self, signal_name, intensity=1.0):
        # 1. Get Weight
        weight = self.signal_weights.get(signal_name, 0.1)
        if weight == 0.1:
            for w_key, w_val in self.signal_weights.items():
                if w_key in signal_name:
                    weight = w_val
                    break

        # 2. Update Score
        self.score += (weight * intensity)

        # 3. Cap the score (0 to 100 for the UI percentage)
        self.score = max(0.0, min(self.score * 10, 100.0))
        return self.score

    def decay(self):
        self.score = max(0.0, self.score - self.decay_rate)
        return self.score

    def get_meaning(self, raw_text):
        """Translates raw detection labels into psychological context."""
        mapping = {
            # Face
            "lips_compressed": "🤐 Holding Back Speech",
            "chin_thrust": "💪 Challenge / Anger",
            "eye_squint": "🤔 Disbelief / Evaluation",
            "eyebrow_flash": "👋 Recognition / Surprise",
            "flushing": "😳 High Stress / Embarrassment",

            # Body
            "arms_crossed": "🛡️ Defensive / Self-Comfort",
            "hand_on_face": "🧠 Cognitive Load / Hiding",
            "shoulder_shrug": "🤷 Doubt / Indifference",
            "security_check": "😟 Anxiety (Checking Pockets)",
            "leaning_away": "🚫 Disengagement",

            # Audio
            "pitch_rise": "📈 Stress / Emotion",
            "hesitation": "⏳ Searching for words",
        }

        key = raw_text.lower().replace(" ", "_")
        for m_key, m_val in mapping.items():
            if m_key in key:
                return m_val

        return raw_text.title()
