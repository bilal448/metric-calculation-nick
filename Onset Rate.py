import librosa

# Load audio
audio_path = "trimmed_output33.wav"
y, sr = librosa.load(audio_path, sr=None)

# Detect onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')

# Get duration
duration = librosa.get_duration(y=y, sr=sr)

# Calculate onset rate (onsets per second)
onset_rate = len(onset_frames) / duration

print(f"Total onsets: {len(onset_frames)}")
print(f"Duration: {duration:.2f} seconds")
print(f"Onset Rate: {onset_rate:.2f} onsets/second")