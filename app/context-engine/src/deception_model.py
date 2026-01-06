# app/context-engine/src/deception_model.py

class DeceptionModel:
    def __init__(self):
        # Weights: How much does each signal contribute to the Deception Score?
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

        self.score = 0.0
        self.decay_rate = 0.05

    def update_score(self, signal_name, intensity=1.0):
        # 1. Get Weight (default to 0.1 for unknown signals)
        weight = self.signal_weights.get(signal_name, 0.1)

        # 2. Update Score
        # We multiply by intensity (e.g., how red was the face?)
        self.score += (weight * intensity)

        # 3. Cap the score (0 to 10)
        self.score = max(0.0, min(self.score, 10.0))

        return self.score

    def decay(self):
        """Lowers score over time (simulating cooling off)."""
        self.score = max(0.0, self.score - self.decay_rate)
        return self.score
