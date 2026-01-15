# "Danceability" describes how suitable a music track is for dancing, based on a combination of musical elements. It is commonly quantified by music analysis
# services like Spotify on a scale of 0.0 to 1.0, where higher values mean a song is easier to dance to

import librosa
import numpy as np


def calculate_danceability_for_db(audio_path):
    """
    Production-ready danceability calculation
    Returns single 0-1 score
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Tempo
        tempo_array, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_array[0]) if isinstance(tempo_array, np.ndarray) else float(tempo_array)
        tempo_score = np.exp(-((tempo - 120) ** 2) / (2 * 40 ** 2))

        # Rhythm stability
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) > 1:
            ibi = np.diff(beat_times)
            rhythm_stability = 1.0 / (1.0 + np.std(ibi) / np.mean(ibi))
        else:
            rhythm_stability = 0.0

        # Beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(beat_frames) > 0:
            beat_strength = np.tanh(np.mean(onset_env[beat_frames]) * 3)
        else:
            beat_strength = 0.0

        # Regularity
        onset_norm = onset_env / np.max(onset_env) if np.max(onset_env) > 0 else onset_env
        autocorr = librosa.autocorrelate(onset_norm, max_size=len(onset_norm) // 2)
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
        regularity = np.max(autocorr[1:]) if len(autocorr) > 1 else 0

        # Combined score
        danceability = (
                0.30 * tempo_score +
                0.25 * rhythm_stability +
                0.25 * beat_strength +
                0.20 * regularity
        )

        return float(danceability)

    except Exception as e:
        print(f"Error: {e}")
        return 0.0


# Usage
danceability = calculate_danceability_for_db("trimmed_output33.wav")
print(f"Danceability: {danceability:.4f}")