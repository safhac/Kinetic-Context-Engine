def detect_psychological_distancing(text):
    # Doc #109: Euphemizing crimes (e.g., 'hurt' instead of 'kill').
    euphemisms = {"hurt": "kill", "take": "steal",
                  "relations": "sex", "touch": "molest"}
    words = text.lower().split()
    return any(w in words for w in euphemisms.keys())


def detect_pronoun_absence(text):
    # Doc #113: Lack of first-person pronouns due to cognitive load.
    pronouns = ["i", "me", "my", "we"]
    words = text.lower().split()
    return not any(p in words for p in pronouns)


def detect_non_contracting_statement(text):
    # Doc #115: Using 'did not' instead of 'didn't' to sound more matter-of-fact.
    formals = ["did not", "could not", "was not", "is not"]
    return any(f in text.lower() for f in formals)


def detect_vocal_pitch_rise(audio_pitch_stream, baseline_pitch):
    # Doc #110: Stress causes vocal muscles to tighten, raising pitch.
    return audio_pitch_stream > (baseline_pitch * 1.2)
