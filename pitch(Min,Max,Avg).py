#it's min max and avg pitch

"""
Pitch Detection with Demucs - Min, Max, Average
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_pitch_stats(audio_path):
    """
    Calculate pitch statistics using Demucs vocal separation

    Returns:
        min_pitch_hz (float): Lowest vocal pitch detected
        max_pitch_hz (float): Highest vocal pitch detected
        avg_pitch_hz (float): Average vocal pitch
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

    # Detect pitch using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        vocals_mono,
        sr=sr,
        fmin=50,
        fmax=2000
    )

    # Filter by confidence threshold
    confident_voiced = f0[voiced_probs > 0.5]
    confident_voiced = confident_voiced[~np.isnan(confident_voiced)]

    if len(confident_voiced) > 0:
        min_pitch_hz = float(np.min(confident_voiced))
        max_pitch_hz = float(np.max(confident_voiced))
        avg_pitch_hz = float(np.mean(confident_voiced))
    else:
        min_pitch_hz = 0.0
        max_pitch_hz = 0.0
        avg_pitch_hz = 0.0

    return min_pitch_hz, max_pitch_hz, avg_pitch_hz


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    min_pitch, max_pitch, avg_pitch = get_pitch_stats(audio_file)

    print(f"min_pitch_hz: {min_pitch:.2f}")
    print(f"max_pitch_hz: {max_pitch:.2f}")
    print(f"avg_pitch_hz: {avg_pitch:.2f}")