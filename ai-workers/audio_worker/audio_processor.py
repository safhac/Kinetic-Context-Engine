import numpy as np
import whisper
import librosa
import torch
import io
import logging

# Load Whisper model once (using 'tiny' or 'base' for CPU speed)
# 'tiny' is very fast but less accurate. 'base' is a good balance.
MODEL_SIZE = "base"
print(f"⏳ Loading Whisper Model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)
print("✅ Whisper Model Loaded.")


class AudioProcessor:
    def __init__(self, buffer_duration=3.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.audio_buffer = np.array([], dtype=np.float32)

        # Baselines
        # Default Hz (Male ~100-150, Female ~170-220)
        self.baseline_pitch = 120.0
        self.pitch_history = []

    def add_audio_chunk(self, audio_bytes):
        """
        Receives raw PCM audio bytes (float32), adds to buffer.
        Returns True if buffer is full and ready for processing.
        """
        # Assume input is raw float32 bytes
        chunk = np.frombuffer(audio_bytes, dtype=np.float32)
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

        return len(self.audio_buffer) >= self.buffer_size

    def process_buffer(self):
        """
        Process the accumulated audio: Transcribe + Pitch.
        Returns: (text, current_pitch)
        """
        if len(self.audio_buffer) == 0:
            return "", 0.0

        audio_data = self.audio_buffer.copy()

        # 1. Transcription (Whisper)
        # Whisper expects raw audio at 16k
        result = model.transcribe(audio_data, fp16=False)  # fp16=False for CPU
        text = result['text'].strip()

        # 2. Pitch Extraction (Librosa)
        # Extract Fundamental Frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )

        # Calculate mean pitch of voiced segments
        current_pitch = 0.0
        voiced_pitches = f0[~np.isnan(f0)]
        if len(voiced_pitches) > 0:
            current_pitch = np.mean(voiced_pitches)
            self._update_baseline(current_pitch)

        # Clear buffer (or keep overlap window if needed)
        self.audio_buffer = np.array([], dtype=np.float32)

        return text, current_pitch

    def _update_baseline(self, pitch):
        """Slowly adapts baseline pitch to the user"""
        self.pitch_history.append(pitch)
        if len(self.pitch_history) > 20:  # Last ~1 minute
            self.pitch_history.pop(0)
        self.baseline_pitch = sum(self.pitch_history) / len(self.pitch_history)
