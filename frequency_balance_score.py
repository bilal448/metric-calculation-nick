# There is no single, standardized measurement called a "frequency balance score." Instead, "frequency balance"
# (or "spectral balance") is an audio engineering term describing how evenly sound energy is distributed across
# the entire audible frequency spectrum (20 Hz to 20 kHz).

import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')


class FrequencyBalanceAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.y = None

    def load_audio(self):
        """Load audio file"""
        print(f"Loading: {self.audio_file}")
        self.y, self.sr = librosa.load(self.audio_file, sr=self.sr)
        duration = len(self.y) / self.sr
        print(f"Duration: {duration:.1f}s | Sample rate: {self.sr} Hz\n")

    def analyze_frequency_balance(self):
        """Analyze spectral balance across frequency bands"""
        print("=" * 70)
        print("FREQUENCY BALANCE ANALYSIS")
        print("=" * 70)

        # Compute STFT
        stft = librosa.stft(self.y, n_fft=4096, hop_length=2048)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Define frequency bands (standard audio engineering bands)
        bands = {
            'Sub-Bass': (20, 60),
            'Bass': (60, 250),
            'Low-Mid': (250, 500),
            'Mid': (500, 2000),
            'High-Mid': (2000, 4000),
            'Presence': (4000, 6000),
            'Brilliance': (6000, 20000)
        }

        # Calculate energy in each band
        band_energies = {}
        band_energies_db = {}

        print("\n[1] ENERGY DISTRIBUTION BY FREQUENCY BAND")
        print("-" * 70)

        total_energy = 0

        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency bins within this band
            band_mask = (freqs >= low_freq) & (freqs < high_freq)

            # Sum energy in this band
            band_energy = np.sum(power[band_mask, :])
            band_energies[band_name] = band_energy
            total_energy += band_energy

            # Convert to dB
            band_energy_db = 10 * np.log10(band_energy + 1e-10)
            band_energies_db[band_name] = band_energy_db

        # Calculate percentages
        print(f"\n{'Band':<15} {'Range':<15} {'Energy (dB)':<15} {'Percentage':<12} {'Bar'}")
        print("-" * 70)

        for band_name, (low_freq, high_freq) in bands.items():
            percentage = (band_energies[band_name] / total_energy) * 100
            energy_db = band_energies_db[band_name]

            # Visual bar
            bar_length = int(percentage / 2)
            bar = '█' * bar_length

            print(f"{band_name:<15} {low_freq:>5}-{high_freq:<6} Hz  {energy_db:>7.1f} dB  {percentage:>6.2f}%  {bar}")

        # Calculate balance metrics
        print("\n[2] BALANCE METRICS")
        print("-" * 70)

        # Normalize energies for statistical analysis
        energy_values = np.array(list(band_energies.values()))
        energy_percentages = (energy_values / total_energy) * 100

        # 1. Coefficient of Variation (lower = more balanced)
        mean_energy = np.mean(energy_percentages)
        std_energy = np.std(energy_percentages)
        cv = std_energy / mean_energy if mean_energy > 0 else 0

        # 2. Balance Score (0-1, higher = more balanced)
        balance_score = 1.0 / (1.0 + cv)

        # 3. Entropy (higher = more evenly distributed)
        # Normalize to probability distribution
        prob_dist = energy_values / np.sum(energy_values)
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        max_entropy = np.log2(len(bands))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy

        print(f"\nCoefficient of Variation: {cv:.3f}")
        print(f"  → Lower is better (0 = perfectly balanced, >1 = very unbalanced)")

        print(f"\nBalance Score: {balance_score:.3f}")
        print(f"  → Scale: 0 (unbalanced) to 1 (perfectly balanced)")

        print(f"\nSpectral Entropy: {normalized_entropy:.3f}")
        print(f"  → Scale: 0 (concentrated) to 1 (evenly distributed)")

        # 4. Specific band ratios (common audio engineering checks)
        print("\n[3] CRITICAL FREQUENCY RATIOS")
        print("-" * 70)

        bass_energy = band_energies['Sub-Bass'] + band_energies['Bass']
        mid_energy = band_energies['Low-Mid'] + band_energies['Mid']
        treble_energy = band_energies['High-Mid'] + band_energies['Presence'] + band_energies['Brilliance']

        total_bmt = bass_energy + mid_energy + treble_energy

        bass_pct = (bass_energy / total_bmt) * 100
        mid_pct = (mid_energy / total_bmt) * 100
        treble_pct = (treble_energy / total_bmt) * 100

        print(f"\nBass (20-250 Hz):    {bass_pct:>6.2f}%")
        print(f"Mids (250-4000 Hz):  {mid_pct:>6.2f}%")
        print(f"Treble (4-20 kHz):   {treble_pct:>6.2f}%")

        # Bass-to-Treble ratio
        bt_ratio = bass_energy / treble_energy if treble_energy > 0 else 0
        bt_ratio_db = 10 * np.log10(bt_ratio) if bt_ratio > 0 else -np.inf

        print(f"\nBass-to-Treble Ratio: {bt_ratio:.2f} ({bt_ratio_db:+.1f} dB)")

        # 5. Spectral centroid and rolloff
        print("\n[4] SPECTRAL CHARACTERISTICS")
        print("-" * 70)

        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        centroid_mean = np.mean(centroid)

        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=0.85)
        rolloff_mean = np.mean(rolloff)

        print(f"\nSpectral Centroid: {centroid_mean:.1f} Hz")
        print(f"  → 'Center of mass' of the spectrum")

        print(f"\nSpectral Rolloff (85%): {rolloff_mean:.1f} Hz")
        print(f"  → Frequency below which 85% of energy is contained")

        # Brightness classification
        if centroid_mean > 3000:
            brightness = "BRIGHT (high-frequency dominant)"
        elif centroid_mean > 1500:
            brightness = "BALANCED (neutral frequency distribution)"
        else:
            brightness = "DARK (low-frequency dominant)"

        print(f"\nBrightness: {brightness}")

        # 6. Overall interpretation
        print("\n[5] INTERPRETATION")
        print("-" * 70)

        if balance_score > 0.8:
            print("\n✓ WELL-BALANCED MIX")
            print("  Energy is evenly distributed across the frequency spectrum.")
            print("  This indicates professional mixing with good spectral balance.")
        elif balance_score > 0.6:
            print("\n○ MODERATELY BALANCED MIX")
            print("  Generally balanced with some frequency bands more emphasized.")
            print("  Typical of genre-specific mixes (e.g., bass-heavy EDM).")
        elif balance_score > 0.4:
            print("\n△ SOMEWHAT UNBALANCED MIX")
            print("  Noticeable emphasis on certain frequency ranges.")
            print("  May benefit from EQ adjustments for better balance.")
        else:
            print("\n✗ UNBALANCED MIX")
            print("  Significant imbalance across the frequency spectrum.")
            print("  Strong emphasis on specific bands; check mixing/mastering.")

        # Specific recommendations
        print("\nSpecific Observations:")

        if bass_pct > 40:
            print("  • BASS-HEAVY: High sub/bass energy (typical in EDM, hip-hop)")
        elif bass_pct < 20:
            print("  • BASS-LIGHT: Low sub/bass energy (may lack warmth/power)")

        if mid_pct < 25:
            print("  • MID-SCOOPED: 'V-shaped' EQ (common in metal, some EDM)")
        elif mid_pct > 45:
            print("  • MID-FORWARD: Strong mid presence (typical in rock, vocals)")

        if treble_pct > 35:
            print("  • TREBLE-HEAVY: Bright, airy mix (modern pop production)")
        elif treble_pct < 20:
            print("  • TREBLE-LIGHT: Darker mix (may lack clarity/air)")

        if abs(bt_ratio_db) < 3:
            print("  • EVEN BASS/TREBLE BALANCE: Well-balanced low/high end")
        elif bt_ratio_db > 3:
            print("  • BASS DOMINANCE: Bass overpowers treble")
        else:
            print("  • TREBLE DOMINANCE: Treble overpowers bass")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "trimmed_output33.wav"  # Change this to your WAV file name

    print("\n" + "=" * 70)
    print("FREQUENCY BALANCE ANALYZER")
    print("=" * 70)
    print("\nAnalyzing spectral energy distribution across audible spectrum")
    print("(20 Hz - 20 kHz)\n")

    # Initialize and run
    analyzer = FrequencyBalanceAnalyzer(AUDIO_FILE)
    analyzer.load_audio()
    analyzer.analyze_frequency_balance()

    #Save the balance score