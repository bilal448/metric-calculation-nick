# Harmonic rhythm is the rate at which chords change in a piece of music,
# creating a background pulse distinct from the melody's rhythm

import librosa
import numpy as np

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Get chroma features (harmonic content)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Detect chord changes (when chroma pattern changes significantly)
chroma_changes = []

for i in range(1, chroma.shape[1]):
    # Cosine distance between consecutive frames
    similarity = np.dot(chroma[:, i], chroma[:, i - 1]) / (
            np.linalg.norm(chroma[:, i]) * np.linalg.norm(chroma[:, i - 1]) + 1e-10
    )

    # Change detected when similarity drops
    if similarity < 0.9:  # Threshold
        chroma_changes.append(i)

# Convert to time
change_times = librosa.frames_to_time(chroma_changes, sr=sr)
duration = librosa.get_duration(y=y, sr=sr)

# Harmonic rhythm = chord changes per second
harmonic_rhythm = len(change_times) / duration

print(f"Harmonic Rhythm: {harmonic_rhythm:.2f} changes/second")
print(f"Total chord changes: {len(change_times)}")