import numpy as np
import parselmouth

# --- LINGUISTIC HELPER FUNCTIONS (Text) ---


def detect_psychological_distancing(text: str) -> bool:
    """Doc #109: Euphemizing crimes or harsh realities."""
    if not text:
        return False
    euphemisms = ["hurt", "take", "relations", "touch", "mess up"]
    words = text.lower().split()
    return any(w in words for w in euphemisms)


def detect_pronoun_absence(text: str) -> bool:
    """Doc #113: Lack of first-person pronouns (I, me, my) due to cognitive load/distancing."""
    if not text:
        return False
    words = text.lower().split()
    if len(words) < 3:
        return False  # Too short to judge

    pronouns = ["i", "i'm", "i've", "i'll",
                "i'd", "me", "my", "we", "mine", "ours"]
    cleaned_words = [w.strip(".,!?") for w in words]

    # Returns True if NONE of the pronouns are found
    return not any(p in cleaned_words for p in pronouns)


def detect_non_contracting_statement(text: str) -> bool:
    """Doc #115: Using 'did not' instead of 'didn't'."""
    if not text:
        return False
    formals = ["did not", "could not", "was not",
               "is not", "do not", "will not"]
    text_lower = text.lower()
    return any(f in text_lower for f in formals)


def detect_over_apologizing(text: str) -> bool:
    """Doc #119: Sudden presence of apologies."""
    if not text:
        return False
    apologies = ["sorry", "apologize", "forgive"]
    count = sum(1 for w in text.lower().split() if w in apologies)
    return count >= 1


def detect_exclusions(text: str) -> bool:
    """Doc #121: Exclusionary qualifiers."""
    if not text:
        return False
    phrases = ["basically", "probably", "i suppose",
               "as far as i know", "to the best of my knowledge"]
    t = text.lower()
    return any(p in t for p in phrases)


# --- PARALINGUISTIC HELPER FUNCTIONS (Audio) ---

def detect_vocal_pitch_rise(current_mean_pitch: float, baseline_pitch: float) -> bool:
    """Doc #110: Stress causes vocal muscles to tighten, raising pitch > 20%."""
    if baseline_pitch <= 0 or current_mean_pitch <= 0:
        return False
    return current_mean_pitch > (baseline_pitch * 1.20)


def detect_rising_intonation(pitch_values):
    """Detects if a segment ends significantly higher than it started (Questioning tone)."""
    if len(pitch_values) < 5:
        return False

    mid = len(pitch_values) // 2
    start_avg = np.mean(pitch_values[:mid])
    end_avg = np.mean(pitch_values[mid:])

    # If end is 15% higher than start
    return end_avg > (start_avg * 1.15)


def detect_shouting(intensity_obj, threshold_db=82):
    """Checks for high intensity (volume) spikes."""
    if not intensity_obj:
        return False
    # Get max intensity in this segment
    max_val = np.max(intensity_obj.values)
    return max_val > threshold_db


# --- MAIN ENTRY POINT ---

def analyze_segment(sound_segment, text=None, baseline_pitch=None):
    """
    Args:
        sound_segment: A parselmouth.Sound object for the current window.
        text: Transcribed text string (Optional).
        baseline_pitch: Float (Hz).
    """
    signals = []

    # 1. AUDIO ANALYSIS
    if sound_segment:
        # Extract Pitch (F0)
        pitch = sound_segment.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        # Filter unvoiced (0 Hz)
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) > 0:
            current_pitch = np.mean(pitch_values)

            # Signal: Pitch Rise (Stress)
            if baseline_pitch and detect_vocal_pitch_rise(current_pitch, baseline_pitch):
                signals.append("vocal_pitch_rise")

            # Signal: Rising Intonation (Uncertainty/Question)
            if detect_rising_intonation(pitch_values):
                signals.append("rising_intonation")

        # Extract Intensity (Volume)
        intensity = sound_segment.to_intensity()
        if detect_shouting(intensity):
            signals.append("high_volume_spike")

    # 2. TEXT ANALYSIS
    if text:
        if detect_psychological_distancing(text):
            signals.append("psychological_distancing")
        if detect_pronoun_absence(text):
            signals.append("pronoun_absence")
        if detect_non_contracting_statement(text):
            signals.append("non_contracting_statement")
        if detect_over_apologizing(text):
            signals.append("over_apologizing")
        if detect_exclusions(text):
            signals.append("exclusion_words")

    return signals
