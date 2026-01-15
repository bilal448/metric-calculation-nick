"""
Pitch variability is the natural fluctuation in the highness or lowness of a sound, crucial for
conveying emotion, emphasis, and meaning in speech, differing from loudness (volume) by being tied to
sound wave frequency, with faster vibrations (higher frequency) creating higher pitches and slower vibrations
(lower frequency) producing lower pitches, impacting communication from questioning tones to identifying anger or excitement.

"""

"""
Pitch Variability using Demucs
Measures natural fluctuation in pitch (F0) over time
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_pitch_variability(audio_path):
    """
    Calculate pitch variability using Demucs vocal separation

    Measures:
    - Standard Deviation (SD): Dispersion of F0 values
    - Coefficient of Variation (CV): Normalized variation (SD/mean)
    - Pitch Range: Max - Min pitch

    Returns:
        pitch_std (float): Standard deviation of pitch in Hz
        pitch_cv (float): Coefficient of variation (normalized)
        pitch_range (float): Range (max - min) in Hz
        avg_pitch (float): Mean pitch in Hz
    """
    # Load Demucs model
    model = get_model('htdemucs')
    model.eval()

    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    # Convert to torch tensor
    if y.ndim == 1:
        waveform = torch.from_numpy(np.stack([y, y]))
    else:
        waveform = torch.from_numpy(y)

    waveform = waveform.float().unsqueeze(0)

    # Separate sources with Demucs
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract vocals (index 3)
    vocals = sources[0, 3].cpu().numpy()

    # Convert to mono
    vocals_mono = np.mean(vocals, axis=0)

    # Detect pitch (F0) using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        vocals_mono,
        sr=sr,
        fmin=50,
        fmax=2000,
        frame_length=2048
    )

    # Filter by confidence threshold
    confident_f0 = f0[voiced_probs > 0.5]
    confident_f0 = confident_f0[~np.isnan(confident_f0)]

    if len(confident_f0) > 1:
        # Standard Deviation
        pitch_std = float(np.std(confident_f0))

        # Mean pitch
        avg_pitch = float(np.mean(confident_f0))

        # Coefficient of Variation (normalized)
        pitch_cv = float(pitch_std / avg_pitch) if avg_pitch > 0 else 0.0

        # Pitch Range (max - min)
        pitch_range = float(np.max(confident_f0) - np.min(confident_f0))
    else:
        pitch_std = 0.0
        pitch_cv = 0.0
        pitch_range = 0.0
        avg_pitch = 0.0

    return pitch_std, pitch_cv, pitch_range, avg_pitch


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    pitch_std, pitch_cv, pitch_range, avg_pitch = get_pitch_variability(audio_file)

    print(f"Pitch Standard Deviation: {pitch_std:.2f} Hz")
    print(f"Pitch Coefficient of Variation: {pitch_cv:.4f}")
    print(f"Pitch Range: {pitch_range:.2f} Hz")
    print(f"Average Pitch: {avg_pitch:.2f} Hz")