# Vocal heaviness, or vocal weight, describes the perceived thickness, richness, or
# buzziness of a voice, determined by how much vocal fold mass vibrates, affecting its
# tonal quality (timbre).

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


class VocalHeavinessAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate vocals"""
        print("=" * 70)
        print("VOCAL HEAVINESS ANALYZER")
        print("=" * 70)
        print(f"\nLoading: {self.audio_file}")

        # Load audio
        audio, sr = librosa.load(self.audio_file, sr=self.sr, mono=False)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.eval()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Separate sources
        print("Separating vocals...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract vocals (index 3)
        self.stems['vocals'] = librosa.to_mono(sources[3].numpy())

        print("✓ Separation complete!\n")

    def calculate_vocal_heaviness(self):
        """Analyze vocal heaviness (thickness, richness, weight)"""
        print("=" * 70)
        print("VOCAL HEAVINESS ANALYSIS")
        print("=" * 70)

        vocals = self.stems['vocals']

        # Check if vocals are present
        vocals_rms = np.sqrt(np.mean(vocals ** 2))
        if vocals_rms < 1e-6:
            print("\n⚠ No significant vocal content detected!")
            print("This may be an instrumental track.\n")
            return

        # 1. LOW-FREQUENCY ENERGY (chest voice, warmth)
        print("\n[1] LOW-FREQUENCY CONTENT (Warmth/Chest Voice)")
        print("-" * 70)

        stft = librosa.stft(vocals, n_fft=4096)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Low frequencies (80-250 Hz) - chest voice
        low_mask = (freqs >= 80) & (freqs < 250)
        low_energy = np.sum(magnitude[low_mask, :])

        # Mid frequencies (250-2000 Hz) - core vocal range
        mid_mask = (freqs >= 250) & (freqs < 2000)
        mid_energy = np.sum(magnitude[mid_mask, :])

        # High frequencies (2000-8000 Hz) - brightness/clarity
        high_mask = (freqs >= 2000) & (freqs < 8000)
        high_energy = np.sum(magnitude[high_mask, :])

        total_energy = low_energy + mid_energy + high_energy

        if total_energy > 0:
            low_percent = (low_energy / total_energy) * 100
            mid_percent = (mid_energy / total_energy) * 100
            high_percent = (high_energy / total_energy) * 100
        else:
            low_percent = mid_percent = high_percent = 0

        print(f"\nLow (80-250 Hz):     {low_percent:>6.2f}%  {'█' * int(low_percent / 2)}")
        print(f"Mid (250-2000 Hz):   {mid_percent:>6.2f}%  {'█' * int(mid_percent / 2)}")
        print(f"High (2-8 kHz):      {high_percent:>6.2f}%  {'█' * int(high_percent / 2)}")

        # 2. SPECTRAL CENTROID (perceived brightness)
        print("\n[2] SPECTRAL CHARACTERISTICS")
        print("-" * 70)

        centroid = librosa.feature.spectral_centroid(y=vocals, sr=self.sr)
        centroid_mean = np.mean(centroid)

        print(f"\nSpectral Centroid: {centroid_mean:.1f} Hz")

        if centroid_mean < 500:
            print("  → VERY DARK/HEAVY vocal tone")
        elif centroid_mean < 800:
            print("  → DARK/RICH vocal tone")
        elif centroid_mean < 1200:
            print("  → WARM/FULL vocal tone")
        elif centroid_mean < 1800:
            print("  → BALANCED vocal tone")
        elif centroid_mean < 2500:
            print("  → BRIGHT vocal tone")
        else:
            print("  → VERY BRIGHT/LIGHT vocal tone")

        # 3. SPECTRAL FLATNESS (noisiness/breathiness)
        flatness = librosa.feature.spectral_flatness(y=vocals)
        flatness_mean = np.mean(flatness)

        print(f"\nSpectral Flatness: {flatness_mean:.3f}")

        if flatness_mean > 0.5:
            print("  → BREATHY/AIRY quality (whisper-like)")
        elif flatness_mean > 0.3:
            print("  → MODERATE breathiness")
        else:
            print("  → TONAL/SOLID quality (full voice)")

        # 4. HARMONIC-TO-NOISE RATIO (vocal clarity)
        print("\n[3] HARMONIC CONTENT (Richness)")
        print("-" * 70)

        # Estimate pitch
        f0 = librosa.yin(vocals, fmin=80, fmax=600, sr=self.sr)
        f0_valid = f0[~np.isnan(f0)]

        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
            print(f"\nFundamental Frequency: {f0_mean:.1f} Hz")

            if f0_mean < 150:
                print("  → BASS/LOW voice (very heavy)")
            elif f0_mean < 200:
                print("  → BARITONE/LOW voice (heavy)")
            elif f0_mean < 250:
                print("  → TENOR/MID voice (moderate)")
            elif f0_mean < 350:
                print("  → ALTO/HIGH voice (lighter)")
            else:
                print("  → SOPRANO/VERY HIGH voice (light)")

        # Calculate odd-to-even harmonic ratio
        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
            odd_energy = 0
            even_energy = 0

            for harmonic in range(1, 10):
                target_freq = f0_mean * harmonic
                if target_freq < self.sr / 2:
                    idx = np.argmin(np.abs(freqs - target_freq))
                    energy = np.mean(magnitude[idx, :])

                    if harmonic % 2 == 1:
                        odd_energy += energy
                    else:
                        even_energy += energy

            if even_energy > 0:
                odd_even_ratio = odd_energy / even_energy
                print(f"\nOdd-to-Even Harmonic Ratio: {odd_even_ratio:.2f}")

                if odd_even_ratio > 2.0:
                    print("  → HOLLOW/NASAL quality (odd harmonics dominant)")
                elif odd_even_ratio > 1.2:
                    print("  → FULL/RICH quality (balanced with odd bias)")
                elif odd_even_ratio > 0.8:
                    print("  → BALANCED harmonic content")
                else:
                    print("  → SMOOTH/MELLOW quality (even harmonics dominant)")

        # 5. RMS ENERGY AND DYNAMICS
        print("\n[4] DYNAMICS AND POWER")
        print("-" * 70)

        rms = librosa.feature.rms(y=vocals)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_db = 20 * np.log10(rms_mean + 1e-10)

        print(f"\nRMS Energy: {rms_mean:.6f} ({rms_db:.1f} dB)")
        print(f"Dynamic Range (std): {rms_std:.6f}")

        # Crest factor (peak-to-average ratio)
        peaks = np.abs(vocals)
        peak_value = np.max(peaks)
        if rms_mean > 0:
            crest_factor = peak_value / rms_mean
            crest_db = 20 * np.log10(crest_factor)

            print(f"Crest Factor: {crest_factor:.2f} ({crest_db:.1f} dB)")

            if crest_db < 6:
                print("  → COMPRESSED/DENSE vocal (heavy, present)")
            elif crest_db < 12:
                print("  → MODERATE dynamics (typical)")
            else:
                print("  → DYNAMIC/EXPRESSIVE vocal (light, varied)")

        # 6. CALCULATE COMPOSITE HEAVINESS SCORE
        print("\n[5] COMPOSITE HEAVINESS SCORE")
        print("-" * 70)

        # Factors contributing to heaviness:
        # 1. Low-frequency content (higher = heavier)
        low_factor = min(low_percent / 30.0, 1.0)  # Normalize to 0-1

        # 2. Dark spectral centroid (lower = heavier)
        centroid_factor = max(0, 1.0 - (centroid_mean - 500) / 1500)
        centroid_factor = min(1.0, max(0.0, centroid_factor))

        # 3. Low flatness = more tonal = heavier
        tonal_factor = 1.0 - flatness_mean

        # 4. Low crest factor = compressed = heavier
        if rms_mean > 0 and peak_value > 0:
            crest_factor = peak_value / rms_mean
            crest_db = 20 * np.log10(crest_factor)
            compression_factor = max(0, 1.0 - (crest_db - 6) / 12)
            compression_factor = min(1.0, max(0.0, compression_factor))
        else:
            compression_factor = 0.5

        # 5. Fundamental frequency (lower = heavier)
        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
            pitch_factor = max(0, 1.0 - (f0_mean - 100) / 300)
            pitch_factor = min(1.0, max(0.0, pitch_factor))
        else:
            pitch_factor = 0.5

        # Weighted combination
        heaviness_score = (
                0.25 * low_factor +
                0.20 * centroid_factor +
                0.15 * tonal_factor +
                0.20 * compression_factor +
                0.20 * pitch_factor
        )

        heaviness_score = min(1.0, max(0.0, heaviness_score))

        print(f"\nComponent Scores:")
        print(f"  Low-Frequency Content:  {low_factor:.3f}")
        print(f"  Dark Tone:              {centroid_factor:.3f}")
        print(f"  Tonal Quality:          {tonal_factor:.3f}")
        print(f"  Compression:            {compression_factor:.3f}")
        print(f"  Low Pitch:              {pitch_factor:.3f}")

        print(f"\n{'=' * 35}")
        print(f"VOCAL HEAVINESS SCORE: {heaviness_score:.3f}")
        print(f"{'=' * 35}")

        # Visual bar
        heaviness_percent = int(heaviness_score * 100)
        bar = '█' * heaviness_percent + '░' * (100 - heaviness_percent)
        print(f"\n[{bar}]")
        print(" Light                                                     Heavy")

        # Rating
        if heaviness_score < 0.2:
            rating = "VERY LIGHT"
            description = "Bright, airy, thin vocal quality"
        elif heaviness_score < 0.4:
            rating = "LIGHT"
            description = "Clear, bright vocal with moderate body"
        elif heaviness_score < 0.6:
            rating = "MODERATE"
            description = "Balanced vocal weight, neither light nor heavy"
        elif heaviness_score < 0.8:
            rating = "HEAVY"
            description = "Rich, thick, full-bodied vocal"
        else:
            rating = "VERY HEAVY"
            description = "Dense, powerful, bass-heavy vocal"

        print(f"\nRating: {rating}")
        print(f"{description}")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = VocalHeavinessAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_vocal_heaviness()
