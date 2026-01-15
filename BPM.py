import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Method 1: Simple BPM estimation
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Fix: Extract the scalar value from the array
tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
# Or simply: tempo = float(tempo)

print(f"Estimated BPM: {tempo:.2f}")
print(f"Number of beats detected: {len(beat_frames)}")

# Convert beat frames to time
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"Beat times (first 10): {beat_times[:10]}")