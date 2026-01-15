# Stereo width refers to the perceived spaciousness or breadth of a sound in the imaginary space between
# left and right speakers, created by sonic differences between the channels, not just panning.

import numpy as np
import librosa
import scipy.signal as signal
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


class StereoWidthAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.left = None
        self.right = None

    def load_audio(self):
        """Load stereo audio file"""
        print(f"Loading: {self.audio_file}")

        # Load as stereo (mono=False)
        audio, self.sr = librosa.load(self.audio_file, sr=self.sr, mono=False)

        # Check if stereo
        if audio.ndim == 1:
            print("⚠ WARNING: Audio is mono. Converting to pseudo-stereo for analysis.")
            self.left = audio
            self.right = audio
            self.is_mono = True
        else:
            self.left = audio[0]
            self.right = audio[1]
            self.is_mono = False

        duration = len(self.left) / self.sr
        print(f"Duration: {duration:.1f}s | Sample rate: {self.sr} Hz")
        print(f"Channels: {'Mono (duplicated)' if self.is_mono else 'Stereo'}\n")

    def calculate_basic_width_metrics(self):
        """Calculate fundamental stereo width metrics"""
        print("=" * 70)
        print("STEREO WIDTH ANALYSIS")
        print("=" * 70)

        print("\n[1] BASIC WIDTH METRICS")
        print("-" * 70)

        # Mid/Side decomposition
        mid = (self.left + self.right) / 2
        side = (self.left - self.right) / 2

        # RMS energy
        left_rms = np.sqrt(np.mean(self.left ** 2))
        right_rms = np.sqrt(np.mean(self.right ** 2))
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))

        # Stereo Width Formula (common in audio tools)
        # Width = (2 * Side) / (Mid + Side)
        if mid_rms + side_rms > 0:
            stereo_width = (2 * side_rms) / (mid_rms + side_rms)
        else:
            stereo_width = 0.0

        # Alternative: Side/Mid ratio
        if mid_rms > 0:
            side_mid_ratio = side_rms / mid_rms
            side_mid_db = 20 * np.log10(side_mid_ratio)
        else:
            side_mid_ratio = 0
            side_mid_db = -np.inf

        print(f"\nStereo Width (0-1):     {stereo_width:.3f}")
        print(f"Side/Mid Ratio:         {side_mid_ratio:.3f} ({side_mid_db:+.1f} dB)")

        # Visual representation
        width_percent = int(stereo_width * 100)
        bar = '█' * (width_percent // 2) + '░' * (50 - width_percent // 2)
        print(f"\nWidth Visualization:    [{bar}] {width_percent}%")
        print("                         [Mono                          Wide]")

        # Interpretation
        print("\nInterpretation:")
        if stereo_width < 0.2:
            print("  → VERY NARROW (Near Mono)")
            print("    Most energy in center, minimal stereo information")
        elif stereo_width < 0.4:
            print("  → NARROW (Center-focused)")
            print("    Predominantly mono with some stereo elements")
        elif stereo_width < 0.6:
            print("  → MODERATE WIDTH (Balanced)")
            print("    Good mix of center and stereo content")
        elif stereo_width < 0.8:
            print("  → WIDE (Spacious)")
            print("    Strong stereo separation, expansive sound")
        else:
            print("  → VERY WIDE (Ultra-spacious)")
            print("    Maximum stereo separation, potential phase issues")

        return stereo_width

    def calculate_inter_channel_differences(self):
        """Analyze differences between L/R channels"""
        print("\n[2] INTER-CHANNEL DIFFERENCE ANALYSIS")
        print("-" * 70)

        # Level differences
        left_rms = np.sqrt(np.mean(self.left ** 2))
        right_rms = np.sqrt(np.mean(self.right ** 2))

        if right_rms > 0:
            level_balance = left_rms / right_rms
            level_balance_db = 20 * np.log10(level_balance)
        else:
            level_balance = 0
            level_balance_db = -np.inf

        print(f"\nLevel Balance (L/R):    {level_balance:.3f} ({level_balance_db:+.2f} dB)")

        if abs(level_balance_db) < 1:
            print("  → Perfectly balanced levels")
        elif abs(level_balance_db) < 3:
            print("  → Well balanced (slight preference to one side)")
        else:
            print("  → Imbalanced (one channel significantly louder)")

        # Spectral differences
        print("\nSpectral Divergence:")

        # Compute spectrograms
        stft_left = librosa.stft(self.left, n_fft=2048)
        stft_right = librosa.stft(self.right, n_fft=2048)

        mag_left = np.abs(stft_left)
        mag_right = np.abs(stft_right)

        # Calculate spectral correlation
        left_flat = mag_left.flatten()
        right_flat = mag_right.flatten()

        if np.std(left_flat) > 1e-6 and np.std(right_flat) > 1e-6:
            spectral_corr, _ = pearsonr(left_flat, right_flat)
        else:
            spectral_corr = 1.0

        spectral_divergence = 1 - spectral_corr

        print(f"  Spectral Correlation:   {spectral_corr:.3f}")
        print(f"  Spectral Divergence:    {spectral_divergence:.3f}")

        if spectral_divergence > 0.5:
            print("  → HIGH divergence: Very different spectral content L/R")
        elif spectral_divergence > 0.2:
            print("  → MODERATE divergence: Some unique content each channel")
        else:
            print("  → LOW divergence: Similar spectral content")

        return spectral_divergence

    def calculate_frequency_dependent_width(self):
        """Calculate stereo width across frequency bands"""
        print("\n[3] FREQUENCY-DEPENDENT WIDTH")
        print("-" * 70)

        bands = [
            ('Sub-Bass', 20, 60),
            ('Bass', 60, 250),
            ('Low-Mid', 250, 500),
            ('Mid', 500, 2000),
            ('High-Mid', 2000, 4000),
            ('Presence', 4000, 6000),
            ('Brilliance', 6000, 20000)
        ]

        print(f"\n{'Band':<12} {'Range':<18} {'Width':<8} {'Bar':<30} {'Description'}")
        print("-" * 70)

        band_widths = []

        for band_name, low_freq, high_freq in bands:
            # Design bandpass filter
            nyquist = self.sr / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.99)

            # Butterworth bandpass filter
            sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')

            # Filter both channels
            left_filtered = signal.sosfilt(sos, self.left)
            right_filtered = signal.sosfilt(sos, self.right)

            # Calculate Mid/Side for this band
            mid_band = (left_filtered + right_filtered) / 2
            side_band = (left_filtered - right_filtered) / 2

            mid_rms = np.sqrt(np.mean(mid_band ** 2))
            side_rms = np.sqrt(np.mean(side_band ** 2))

            # Width for this band
            if mid_rms + side_rms > 0:
                band_width = (2 * side_rms) / (mid_rms + side_rms)
            else:
                band_width = 0.0

            band_widths.append(band_width)

            # Visual bar
            bar_length = int(band_width * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)

            # Description
            if band_width < 0.3:
                desc = "Mono/Center"
            elif band_width < 0.5:
                desc = "Narrow"
            elif band_width < 0.7:
                desc = "Moderate"
            else:
                desc = "Wide"

            print(f"{band_name:<12} {low_freq:>5}-{high_freq:<8} Hz  {band_width:.2f}   {bar:<30} {desc}")

        # Analyze pattern
        print("\nWidth Profile:")
        bass_width = np.mean(band_widths[:2])  # Sub-bass + Bass
        mid_width = np.mean(band_widths[2:4])  # Low-mid + Mid
        high_width = np.mean(band_widths[4:])  # High-mid + Presence + Brilliance

        print(f"  Bass (20-250 Hz):      {bass_width:.2f}")
        print(f"  Mids (250-2000 Hz):    {mid_width:.2f}")
        print(f"  Highs (2-20 kHz):      {high_width:.2f}")

        if bass_width < mid_width < high_width:
            print("\n  → STANDARD PROFILE: Narrow bass → Wide highs")
            print("    Typical professional mix (mono bass, stereo highs)")
        elif bass_width > mid_width and bass_width > high_width:
            print("\n  → WIDE BASS: Unusual - may cause phase issues")
        elif high_width < mid_width:
            print("\n  → NARROW HIGHS: Conservative width, good mono compatibility")
        else:
            print("\n  → CUSTOM PROFILE: Non-standard width distribution")

        return band_widths

    def calculate_perceived_width_score(self):
        """Calculate perceptual stereo width score"""
        print("\n[4] PERCEIVED WIDTH SCORE")
        print("-" * 70)

        # Multiple factors contribute to perceived width

        # 1. Basic stereo width
        mid = (self.left + self.right) / 2
        side = (self.left - self.right) / 2
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))

        if mid_rms + side_rms > 0:
            basic_width = (2 * side_rms) / (mid_rms + side_rms)
        else:
            basic_width = 0.0

        # 2. Spectral diversity between channels
        stft_left = librosa.stft(self.left, n_fft=2048)
        stft_right = librosa.stft(self.right, n_fft=2048)
        mag_left = np.abs(stft_left).flatten()
        mag_right = np.abs(stft_right).flatten()

        if np.std(mag_left) > 1e-6 and np.std(mag_right) > 1e-6:
            spectral_corr, _ = pearsonr(mag_left, mag_right)
        else:
            spectral_corr = 1.0

        spectral_diversity = 1 - spectral_corr

        # 3. High-frequency width (most important for perceived width)
        nyquist = self.sr / 2
        sos = signal.butter(4, [4000 / nyquist, 0.99], btype='band', output='sos')
        left_highs = signal.sosfilt(sos, self.left)
        right_highs = signal.sosfilt(sos, self.right)

        mid_highs = (left_highs + right_highs) / 2
        side_highs = (left_highs - right_highs) / 2

        mid_highs_rms = np.sqrt(np.mean(mid_highs ** 2))
        side_highs_rms = np.sqrt(np.mean(side_highs ** 2))

        if mid_highs_rms + side_highs_rms > 0:
            high_width = (2 * side_highs_rms) / (mid_highs_rms + side_highs_rms)
        else:
            high_width = 0.0

        # 4. Inter-aural time differences (ITD) simulation
        # Cross-correlation to detect time delays
        correlation = signal.correlate(self.left, self.right, mode='valid')
        lag = np.argmax(np.abs(correlation)) - len(self.right) + 1
        time_delay = abs(lag) / self.sr * 1000  # in milliseconds

        # ITD contribution (up to ~1ms is perceptually significant)
        itd_factor = min(time_delay / 1.0, 1.0)

        # Weighted combination for perceived width
        # High frequencies and spectral diversity are most important
        perceived_width = (
                0.25 * basic_width +
                0.15 * spectral_diversity +
                0.40 * high_width +
                0.20 * itd_factor
        )

        perceived_width = min(1.0, max(0.0, perceived_width))

        print(f"\nComponent Scores:")
        print(f"  Basic Width:           {basic_width:.3f}")
        print(f"  Spectral Diversity:    {spectral_diversity:.3f}")
        print(f"  High-Frequency Width:  {high_width:.3f}")
        print(f"  Inter-aural Delay:     {itd_factor:.3f} ({time_delay:.2f} ms)")

        print(f"\n{'=' * 30}")
        print(f"PERCEIVED WIDTH SCORE: {perceived_width:.3f}")
        print(f"{'=' * 30}")

        # Rating
        perceived_percent = int(perceived_width * 100)
        bar = '█' * perceived_percent + '░' * (100 - perceived_percent)
        print(f"\n[{bar}]")

        if perceived_width < 0.2:
            rating = "VERY NARROW"
            description = "Mono-like presentation, minimal spatial depth"
        elif perceived_width < 0.4:
            rating = "NARROW"
            description = "Limited stereo image, mostly center-focused"
        elif perceived_width < 0.6:
            rating = "MODERATE"
            description = "Good stereo presence, balanced spatial presentation"
        elif perceived_width < 0.8:
            rating = "WIDE"
            description = "Expansive soundstage, strong spatial separation"
        else:
            rating = "VERY WIDE"
            description = "Maximum spaciousness, immersive stereo field"

        print(f"\nRating: {rating}")
        print(f"{description}")

        return perceived_width

    def analyze_spatial_distribution(self):
        """Analyze how sound is distributed in the stereo field"""
        print("\n[5] SPATIAL DISTRIBUTION ANALYSIS")
        print("-" * 70)

        # Analyze panning distribution using intensity differences
        # Create time windows
        window_size = int(0.5 * self.sr)  # 500ms windows
        hop_size = window_size // 2

        pan_positions = []

        for i in range(0, len(self.left) - window_size, hop_size):
            left_window = self.left[i:i + window_size]
            right_window = self.right[i:i + window_size]

            left_energy = np.sum(left_window ** 2)
            right_energy = np.sum(right_window ** 2)

            # Calculate pan position (-1 = full left, 0 = center, +1 = full right)
            total_energy = left_energy + right_energy
            if total_energy > 0:
                pan_position = (right_energy - left_energy) / total_energy
            else:
                pan_position = 0

            pan_positions.append(pan_position)

        pan_positions = np.array(pan_positions)

        # Statistics
        mean_pan = np.mean(pan_positions)
        std_pan = np.std(pan_positions)

        # Distribution across field
        left_percentage = np.sum(pan_positions < -0.33) / len(pan_positions) * 100
        center_percentage = np.sum(np.abs(pan_positions) <= 0.33) / len(pan_positions) * 100
        right_percentage = np.sum(pan_positions > 0.33) / len(pan_positions) * 100

        print(f"\nSpatial Distribution:")
        print(f"  Left (L):      {left_percentage:>6.1f}%  {'█' * int(left_percentage / 2)}")
        print(f"  Center (C):    {center_percentage:>6.1f}%  {'█' * int(center_percentage / 2)}")
        print(f"  Right (R):     {right_percentage:>6.1f}%  {'█' * int(right_percentage / 2)}")

        print(f"\nBalance:")
        print(f"  Mean Position: {mean_pan:+.3f}")

        if abs(mean_pan) < 0.05:
            print("  → Perfectly centered balance")
        elif abs(mean_pan) < 0.15:
            print("  → Well-balanced (slight bias)")
        else:
            bias = "left" if mean_pan < 0 else "right"
            print(f"  → Biased toward {bias} channel")

        print(f"\n  Width Variability: {std_pan:.3f}")

        if std_pan > 0.5:
            print("  → High variability: Dynamic panning/width changes")
        elif std_pan > 0.25:
            print("  → Moderate variability: Some spatial movement")
        else:
            print("  → Low variability: Static stereo image")

    def generate_summary(self):
        """Generate comprehensive summary"""
        print("\n" + "=" * 70)
        print("STEREO WIDTH SUMMARY")
        print("=" * 70)

        # Recalculate key metrics for summary
        mid = (self.left + self.right) / 2
        side = (self.left - self.right) / 2
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))

        if mid_rms + side_rms > 0:
            stereo_width = (2 * side_rms) / (mid_rms + side_rms)
        else:
            stereo_width = 0.0

        # High-frequency width
        nyquist = self.sr / 2
        sos = signal.butter(4, [4000 / nyquist, 0.99], btype='band', output='sos')
        left_highs = signal.sosfilt(sos, self.left)
        right_highs = signal.sosfilt(sos, self.right)
        mid_highs = (left_highs + right_highs) / 2
        side_highs = (left_highs - right_highs) / 2
        mid_highs_rms = np.sqrt(np.mean(mid_highs ** 2))
        side_highs_rms = np.sqrt(np.mean(side_highs ** 2))

        if mid_highs_rms + side_highs_rms > 0:
            high_width = (2 * side_highs_rms) / (mid_highs_rms + side_highs_rms)
        else:
            high_width = 0.0

        print(f"\nKey Metrics:")
        print(f"  Overall Width:         {stereo_width:.3f}")
        print(f"  High-Frequency Width:  {high_width:.3f}")
        print(f"  Side/Mid Energy:       {side_rms / mid_rms if mid_rms > 0 else 0:.3f}")

        print(f"\nMix Characteristics:")

        # Determine mix style based on width
        if stereo_width < 0.3 and high_width < 0.4:
            print("  → MONO/CENTER-DOMINANT MIX")
            print("    Limited stereo information, mono-compatible but narrow")
            print("    Typical of: Old recordings, mono-focused productions")
        elif stereo_width < 0.5 and high_width < 0.6:
            print("  → CONSERVATIVE STEREO MIX")
            print("    Moderate width with good mono compatibility")
            print("    Typical of: Classical, jazz, acoustic music")
        elif stereo_width < 0.7 and high_width < 0.8:
            print("  → WIDE STEREO MIX")
            print("    Strong stereo presence with spatial depth")
            print("    Typical of: Modern pop, rock, electronic music")
        else:
            print("  → ULTRA-WIDE STEREO MIX")
            print("    Maximum spatial separation and immersion")
            print("    Typical of: Electronic music, ambient, experimental")

        # Production recommendations
        print(f"\nProduction Notes:")

        if stereo_width > 0.8:
            print("  ⚠ Check mono compatibility - very wide mixes may collapse")

        if high_width < 0.3:
            print("  • Conservative high-end width - good for broadcast/streaming")
        elif high_width > 0.7:
            print("  • Wide high-end - creates spacious, airy feel")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "trimmed_output33.wav"  # Change this to your WAV file name

    print("\n" + "=" * 70)
    print("STEREO WIDTH ANALYZER")
    print("=" * 70)
    print("\nAnalyzing perceived spaciousness and breadth of stereo field")
    print("Measures sonic differences between channels, not just panning")
    print()

    # Initialize and run
    analyzer = StereoWidthAnalyzer(AUDIO_FILE)
    analyzer.load_audio()
    analyzer.calculate_basic_width_metrics()
    analyzer.calculate_inter_channel_differences()
    analyzer.calculate_frequency_dependent_width()
    analyzer.calculate_perceived_width_score()
    analyzer.analyze_spatial_distribution()
    analyzer.generate_summary()