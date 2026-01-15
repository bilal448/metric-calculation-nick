# The "vocal energy ratio" generally refers to specific quantitative measures used in phonetics
# and acoustics to analyze voice quality and production efficiency, primarily the Singing Power Ratio
# (SPR) or the Harmonics-to-Noise Ratio (HNR).

"""
Vocal Energy Ratio using Demucs
Calculates Singing Power Ratio (SPR) and Harmonics-to-Noise Ratio (HNR) from separated vocals
"""

import torch
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def get_vocal_energy_ratio(audio_path):
    """
    Calculate vocal energy ratio using Demucs separation
    Returns SPR and HNR metrics

    Returns:
        spr (float): Singing Power Ratio (2-4 kHz / 0-2 kHz energy)
        hnr (float): Harmonics-to-Noise Ratio in dB
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

    # Separate sources
    with torch.no_grad():
        sources = apply_model(model, waveform, device='cpu', split=True, overlap=0.25)

    # Extract vocals (index 3)
    vocals = sources[0, 3].cpu().numpy()

    # Convert to mono for analysis
    vocals_mono = np.mean(vocals, axis=0)

    # 1. Calculate Singing Power Ratio (SPR)
    # SPR = Energy in 2-4 kHz / Energy in 0-2 kHz
    stft = librosa.stft(vocals_mono, n_fft=2048, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Energy in 2-4 kHz (singing formant region)
    mask_high = (freqs >= 2000) & (freqs <= 4000)
    energy_high = np.mean(np.abs(stft[mask_high, :]) ** 2)

    # Energy in 0-2 kHz
    mask_low = (freqs >= 0) & (freqs <= 2000)
    energy_low = np.mean(np.abs(stft[mask_low, :]) ** 2)

    spr = float(energy_high / (energy_low + 1e-10))

    # 2. Calculate Harmonics-to-Noise Ratio (HNR)
    # Frame-based HNR calculation
    frame_length = 2048
    hop_length = 512

    hnr_values = []
    for i in range(0, len(vocals_mono) - frame_length, hop_length):
        frame = vocals_mono[i:i + frame_length]

        # Autocorrelation method for HNR
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Find first peak (fundamental period)
        if len(autocorr) > 1:
            # Skip first value (always max)
            peak_idx = np.argmax(autocorr[20:]) + 20  # Skip very short periods

            if peak_idx > 0 and autocorr[0] > 0:
                # HNR = 10 * log10(periodic_energy / noise_energy)
                periodic_power = autocorr[peak_idx]
                total_power = autocorr[0]
                noise_power = total_power - periodic_power

                if noise_power > 0:
                    hnr_frame = 10 * np.log10(periodic_power / noise_power)
                    hnr_values.append(hnr_frame)

    hnr = float(np.mean(hnr_values)) if hnr_values else 0.0

    return spr, hnr


def get_vocal_energy_ratio_simple(audio_path):
    """
    Simplified version - returns single combined metric
    """
    spr, hnr = get_vocal_energy_ratio(audio_path)

    # Normalize and combine (higher is better for both)
    # SPR typically ranges 0.1-2.0, HNR typically 5-25 dB
    spr_normalized = min(spr / 2.0, 1.0)
    hnr_normalized = min(hnr / 25.0, 1.0)

    vocal_energy_ratio = float(0.5 * spr_normalized + 0.5 * hnr_normalized)

    return vocal_energy_ratio


# Usage
if __name__ == '__main__':
    audio_file = "trimmed_output33.wav"

    # Full metrics
    spr, hnr = get_vocal_energy_ratio(audio_file)
    print(f"Singing Power Ratio (SPR): {spr:.4f}")
    print(f"Harmonics-to-Noise Ratio (HNR): {hnr:.2f} dB")

    # Simple combined metric
    ratio = get_vocal_energy_ratio_simple(audio_file)
    print(f"Vocal Energy Ratio (combined): {ratio:.4f}")