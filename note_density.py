# The term "note density" in music refers to the average number of notes per unit of time (e.g., notes per second)
# in a musical piece. It is a key factor in determining the perceived tempo and texture of the music.

"""
Note Density Calculation
Average number of notes per unit of time (notes per second)
"""

import librosa
import numpy as np


def get_note_density(audio_path, sr):
    """
    Calculate note density (notes per second) using onset detection

    Args:
        audio_path: Path to audio file
        sr: Sample rate

    Returns:
        note_density (float): Average notes per second
        total_onsets (int): Total number of detected notes/onsets
        duration (float): Audio duration in seconds
        onset_times (list): Times of each detected note/onset
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Get duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Detect onsets (note attacks)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)

    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    # Calculate note density
    total_onsets = len(onset_times)
    note_density = total_onsets / duration if duration > 0 else 0.0

    return float(note_density), total_onsets, float(duration), onset_times.tolist()

density, total, duration, times = get_note_density('trimmed_output33.wav',None)
print(f"Note Density: {density:.2f} notes/second")