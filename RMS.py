import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# CORRECT: True overall RMS energy
rms_overall = np.sqrt(np.mean(y**2))
print(f"Overall RMS energy: {rms_overall}")

# For temporal analysis (RMS over time):
rms_frames = librosa.feature.rms(y=y)
print(f"RMS per frame: {rms_frames}")

