import numpy as np
import whisper
import librosa

MODEL_SIZE = "base"
print(f"⏳ Loading Whisper Model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)
print("✅ Whisper Model Loaded.")


class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.baseline_pitch = 120.0

    def process_array(self, audio_data):
        """
        Process the loaded audio array directly.
        """
        if len(audio_data) == 0:
            return "", 0.0

        # 1. Transcription (Whisper)
        result = model.transcribe(audio_data, fp16=False)
        text = result['text'].strip()

        # 2. Pitch Extraction (Librosa)
        f0, _, _ = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )

        current_pitch = 0.0
        voiced_pitches = f0[~np.isnan(f0)]
        if len(voiced_pitches) > 0:
            current_pitch = float(np.mean(voiced_pitches))

        return text, current_pitch
