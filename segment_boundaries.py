# Segment boundaries in sound are detected by identifying points in an audio signal where acoustic characteristics
# change significantly. This process, known as audio segmentation, can be achieved manually or automatically using
# various signal processing and machine learning techniques.

import librosa

audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Compute chroma features
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Detect boundaries using Laplacian segmentation
boundaries = librosa.segment.agglomerative(chroma, k=10)

# Convert frames to time
boundary_times = librosa.frames_to_time(boundaries, sr=sr)

print(f"Segment boundaries at: {boundary_times}")
print(f"Number of segments: {len(boundary_times) + 1}")