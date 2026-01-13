from typing import List, Optional
import parselmouth
import numpy as np


def get_active_verbal_signals(text: str, current_pitch: float = None, baseline_pitch: float = None) -> List[str]:
    """
    Main entry point for Verbal/Audio Worker.
    Args:
        text: The transcribed text segment (e.g., from Whisper).
        current_pitch: The fundamental frequency (F0) of the current segment.
        baseline_pitch: The user's average baseline pitch.
    """
    signals = []

    # 1. Linguistic Analysis (Text)
    if text:
        if detect_psychological_distancing(text):
            signals.append("psychological_distancing")

        if detect_pronoun_absence(text):
            signals.append("pronoun_absence")

        if detect_non_contracting_statement(text):
            signals.append("non_contracting_statement")

        if detect_over_apologizing(text):
            signals.append("over_apologizing")

    # 2. Paralinguistic Analysis (Audio Features)
    if current_pitch and baseline_pitch:
        if detect_vocal_pitch_rise(current_pitch, baseline_pitch):
            signals.append("vocal_pitch_rise")

    return signals

# --- Linguistic Helpers ---


def detect_psychological_distancing(text: str) -> bool:
    # [cite_start]Doc #109: Euphemizing crimes (e.g., 'hurt' instead of 'kill')[cite: 540, 541].
    # Logic: Checking for specific softened words in context of severe actions.
    euphemisms = {
        "hurt": "kill",
        "take": "steal",
        "relations": "sex",
        "touch": "molest"
    }
    words = text.lower().split()
    return any(w in words for w in euphemisms.keys())


def detect_pronoun_absence(text: str) -> bool:
    # [cite_start]Doc #113: Lack of first-person pronouns due to cognitive load[cite: 556, 557].
    # Logic: Deceptive statements often drop "I" to dissociate from the act.
    # Note: Requires a sentence of sufficient length to be valid (e.g., > 3 words).
    words = text.lower().split()
    if len(words) < 3:
        return False

    pronouns = ["i", "i'm", "i've", "i'll", "i'd",
                "me", "my", "we", "we're", "mine", "ours"]

    cleaned_words = [w.strip(".,!?") for w in words]
    return not any(p in cleaned_words for p in pronouns)


def detect_non_contracting_statement(text: str) -> bool:
    # [cite_start]Doc #115: Using 'did not' instead of 'didn't' to sound more matter-of-fact[cite: 565, 566].
    formals = ["did not", "could not", "was not",
               "is not", "do not", "will not"]
    text_lower = text.lower()
    return any(f in text_lower for f in formals)


def detect_over_apologizing(text: str) -> bool:
    # [cite_start]Doc #119: Sudden presence of apologies for lack of detail/memory[cite: 579, 580].
    apologies = ["sorry", "apologize", "forgive me"]
    count = sum(1 for w in text.lower().split() if w in apologies)
    return count >= 1

# --- Paralinguistic Helpers ---


def detect_vocal_pitch_rise(current_pitch: float, baseline_pitch: float) -> bool:
    # [cite_start]Doc #110: Stress causes vocal muscles to tighten, raising pitch[cite: 545, 546].
    # Threshold: 20% increase over baseline is a significant stress indicator.
    if baseline_pitch <= 0:
        return False
    return current_pitch > (baseline_pitch * 1.2)


def analyze_pitch_gestures(audio_path):
    """
    Detects 'gestures' in how words are spoken.
    """
    sound = parselmouth.Sound(audio_path)

    # Extract Pitch (F0)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']

    # Filter out 0s (silence/unvoiced)
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) == 0:
        return []

    signals = []

    # --- Gesture 1: Rising Intonation (Question/Uncertainty) ---
    # Compare average of first half vs second half
    mid_point = len(pitch_values) // 2
    start_avg = np.mean(pitch_values[:mid_point])
    end_avg = np.mean(pitch_values[mid_point:])

    if end_avg > start_avg * 1.2:  # 20% rise
        signals.append("rising_intonation")

    # --- Gesture 2: High Energy / Shouting ---
    intensity = sound.to_intensity()
    max_intensity = np.max(intensity.values)

    if max_intensity > 80:  # dB threshold
        signals.append("high_volume_spike")

    return signals
