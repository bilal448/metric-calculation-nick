# Bass timbre refers to the unique sound quality (or tone color) of low-frequency sounds, characterized by depth, fullness, richness, and warmth,
# often felt as a rumble or powerful foundation in music, distinct from mere pitch. It's shaped by factors like instrument construction,
# strings, playing style (plucked, slapped, bowed), and effects (EQ, distortion), creating textures from deep and smooth (like a velvety bass voice) to aggressive and punchy (like a synth bass).

"""
Bass Timbre Type using Demucs
Analyzes the tonal quality/color of bass frequencies
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_bass_timbre_type(audio_path):
    """
    Calculate bass timbre characteristics using Demucs

    Analyzes:
    - Spectral Centroid (brightness)
    - Spectral Rolloff (energy distribution)
    - Roughness (harmonic complexity)

    Returns:
        bass_timbre_type (str): Classification of bass timbre
        spectral_centroid (float): Center of mass of spectrum
        spectral_rolloff (float): Frequency rolloff point
        roughness (float): Harmonic complexity measure
    """
    # Load Demucs model
    model = get_model('htdemucs')
    model.eval()

    # Load audio
    y, sr = librosa.load(audio_path, sr=44100, mono=False)

    if y.ndim == 1:
        waveform = torch.from_numpy(np.stack([y, y]))
    else:
        waveform = torch.from_numpy(y)

    waveform = waveform.float().unsqueeze(0)

    # Separate sources with Demucs
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract bass track (index 1)
    bass = sources[0, 1].cpu().numpy()
    bass_mono = np.mean(bass, axis=0)

    # 1. Spectral Centroid (brightness/tone color)
    spectral_centroid = librosa.feature.spectral_centroid(y=bass_mono, sr=sr, hop_length=512)
    centroid_mean = float(np.mean(spectral_centroid))

    # 2. Spectral Rolloff (frequency distribution)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=bass_mono, sr=sr, hop_length=512, roll_percent=0.85)
    rolloff_mean = float(np.mean(spectral_rolloff))

    # 3. Spectral Flatness (roughness/harmonic complexity)
    spectral_flatness = librosa.feature.spectral_flatness(y=bass_mono, hop_length=512)
    flatness_mean = float(np.mean(spectral_flatness))

    # 4. Zero Crossing Rate (texture)
    zcr = librosa.feature.zero_crossing_rate(bass_mono, hop_length=512)
    zcr_mean = float(np.mean(zcr))

    # Classify bass timbre type based on characteristics
    if centroid_mean < 150:
        if flatness_mean < 0.1:
            bass_timbre_type = "deep_smooth"  # Deep, velvety, warm
        else:
            bass_timbre_type = "deep_rough"  # Deep but textured
    elif centroid_mean < 250:
        if flatness_mean < 0.1:
            bass_timbre_type = "rich_full"  # Rich, full, warm
        else:
            bass_timbre_type = "punchy"  # Punchy, aggressive
    else:
        if flatness_mean < 0.1:
            bass_timbre_type = "bright_warm"  # Bright but warm
        else:
            bass_timbre_type = "aggressive"  # Aggressive, distorted

    return bass_timbre_type, centroid_mean, rolloff_mean, flatness_mean


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    timbre_type, centroid, rolloff, roughness = get_bass_timbre_type(audio_file)

    print(f"Bass Timbre Type: {timbre_type}")
    print(f"Spectral Centroid: {centroid:.2f} Hz")
    print(f"Spectral Rolloff: {rolloff:.2f} Hz")
    print(f"Spectral Flatness (Roughness): {roughness:.4f}")