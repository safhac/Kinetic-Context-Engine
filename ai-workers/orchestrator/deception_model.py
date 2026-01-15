import json
import os
from shared.schemas import GestureSignal


class DeceptionModel:
    def __init__(self):
        # --- YOUR ORIGINAL WEIGHTS (Restored) ---
        # 0.0 = Neutral, 1.0 = High Stress/Deception Indicator
        self.signal_weights = {
            # --- FACE (Visual) ---
            "chin_thrust": 0.6,
            "lips_compressed": 0.5,
            "eyebrow_flash": 0.3,
            "eye_squint": 0.4,
            "flushing": 0.9,       # High: Physiological response hard to fake
            "nodding_yes": 0.1,
            "head_down": 0.6,      # Shame/Guilt indicator
            "confirmation_glance": 0.7,

            # --- BODY (Pose) ---
            "single_shrug": 0.5,
            "double_shrug": 0.4,
            "arms_crossed": 0.2,   # Often just comfort, low weight
            "hand_raise": 0.1,
            "steepling": -0.3,     # Confidence (reduces suspicion score)
            "hand_on_face": 0.6,   # Cognitive load / hiding
            "hushing": 0.7,        # Subconscious desire to stop speaking
            "security_check": 0.5,  # Patting pockets (anxiety)
            "foot_withdrawal": 0.6,  # Flight response

            # --- AUDIO (Verbal & Paralinguistic) ---
            "psychological_distancing": 0.8,  # "Did not" vs "Didn't"
            "pronoun_absence": 0.5,          # "Went to store" vs "I went..."
            "vocal_pitch_rise": 0.7,         # Physiological stress
            "non_contracting_statement": 0.4,
            "over_apologizing": 0.5
        }

        # State State
        self.score = 0.0
        self.decay_rate = 0.05

    def analyze(self, signal: GestureSignal) -> float:
        """
        The Bridge: Adapts the new Orchestrator call to your original logic.
        """
        # 1. Decay the score slightly with every new event (cooling off)
        self.decay()

        # 2. Normalize the input text to match dictionary keys
        # Workers might send "Chin Thrust", we need "chin_thrust"
        key = signal.text.lower().replace(" ", "_")

        # 3. Use your original update logic
        # We use the AI confidence (0.0 - 1.0) as the 'intensity'
        new_score = self.update_score(key, intensity=signal.confidence)

        return new_score

    def update_score(self, signal_name, intensity=1.0):
        # 1. Get Weight (default to 0.1 for unknown signals)
        # Using partial matching keys if exact match fails
        weight = self.signal_weights.get(signal_name, 0.1)

        # If direct match failed, try substring match (e.g. "strong_chin_thrust" -> "chin_thrust")
        if weight == 0.1:
            for w_key, w_val in self.signal_weights.items():
                if w_key in signal_name:
                    weight = w_val
                    break

        # 2. Update Score
        self.score += (weight * intensity)

        # 3. Cap the score (0 to 100 for the UI percentage)
        # Your original was 0-10, but the UI expects %, so we scale up x10
        self.score = max(0.0, min(self.score * 10, 100.0))

        return self.score

    def decay(self):
        """Lowers score over time."""
        self.score = max(0.0, self.score - self.decay_rate)
        return self.score
