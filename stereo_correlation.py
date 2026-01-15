# Stereo correlation measures the phase and level relationship between left (L) and right (R) audio channels,
# indicating how much information they share, from +1 (identical signals) to -1 (180° out of phase), with 0
# meaning no shared info, affecting perceived width and mono compatibility.

import numpy as np
import librosa
import scipy.signal as signal
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


class StereoCorrelationAnalyzer:
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

    def calculate_correlation(self):
        """Calculate overall stereo correlation"""
        print("=" * 70)
        print("STEREO CORRELATION ANALYSIS")
        print("=" * 70)

        # Overall correlation using Pearson correlation coefficient
        correlation, _ = pearsonr(self.left, self.right)

        print("\n[1] OVERALL STEREO CORRELATION")
        print("-" * 70)
        print(f"\nCorrelation Coefficient: {correlation:+.4f}")

        # Visual representation
        bar_length = 50
        position = int((correlation + 1) / 2 * bar_length)
        bar = ' ' * position + '|'

        print(f"\n-1.0 {bar:50s} +1.0")
        print("  ↑                                                    ↑")
        print("Phase                                              Mono")
        print("Issues                                          (Identical)")

        # Interpretation
        print("\nInterpretation:")
        if correlation > 0.95:
            print("  → NEAR MONO: Channels are almost identical")
            print("    Very narrow stereo image, limited spatial information")
        elif correlation > 0.7:
            print("  → NARROW STEREO: Strong correlation between channels")
            print("    Some stereo width but mostly center-focused")
        elif correlation > 0.3:
            print("  → MODERATE STEREO: Good balance of shared/unique content")
            print("    Typical of most commercial mixes with clear stereo image")
        elif correlation > 0.0:
            print("  → WIDE STEREO: Low correlation, significant channel independence")
            print("    Wide stereo field, may have mono compatibility concerns")
        elif correlation > -0.3:
            print("  → VERY WIDE/UNCORRELATED: Minimal shared information")
            print("    Extreme stereo width, potential phase issues")
        else:
            print("  → PHASE PROBLEMS: Negative correlation detected!")
            print("    ⚠ Serious phase issues - will cause cancellation in mono")

    def calculate_mid_side_analysis(self):
        """Analyze Mid/Side components"""
        print("\n[2] MID/SIDE ANALYSIS")
        print("-" * 70)

        # Mid (mono sum): M = (L + R) / 2
        mid = (self.left + self.right) / 2

        # Side (stereo difference): S = (L - R) / 2
        side = (self.left - self.right) / 2

        # Calculate RMS energy
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))
        total_rms = np.sqrt(np.mean(self.left ** 2) + np.mean(self.right ** 2))

        # Mid/Side ratio
        if side_rms > 0:
            ms_ratio = mid_rms / side_rms
            ms_ratio_db = 20 * np.log10(ms_ratio)
        else:
            ms_ratio = np.inf
            ms_ratio_db = np.inf

        # Energy percentages
        total_energy = mid_rms ** 2 + side_rms ** 2
        mid_percent = (mid_rms ** 2 / total_energy) * 100 if total_energy > 0 else 0
        side_percent = (side_rms ** 2 / total_energy) * 100 if total_energy > 0 else 0

        print(f"\nMid (Center) Energy:   {mid_rms:.6f} ({mid_percent:.1f}%)")
        print(f"Side (Stereo) Energy:  {side_rms:.6f} ({side_percent:.1f}%)")
        print(f"\nMid/Side Ratio: {ms_ratio:.2f} ({ms_ratio_db:+.1f} dB)")

        # Visual bars
        mid_bar = '█' * int(mid_percent / 2)
        side_bar = '█' * int(side_percent / 2)

        print(f"\nMid:  {mid_bar}")
        print(f"Side: {side_bar}")

        # Stereo width calculation
        if mid_rms + side_rms > 0:
            stereo_width = (2 * side_rms) / (mid_rms + side_rms)
        else:
            stereo_width = 0

        print(f"\nStereo Width: {stereo_width:.3f}")
        print("  (0 = mono, 1 = maximum stereo width)")

        # Interpretation
        if stereo_width < 0.2:
            print("  → Very narrow stereo image")
        elif stereo_width < 0.5:
            print("  → Moderate stereo width (typical for most mixes)")
        elif stereo_width < 0.8:
            print("  → Wide stereo image")
        else:
            print("  → Extremely wide stereo (may have phase issues)")

    def calculate_frequency_band_correlation(self):
        """Calculate correlation in different frequency bands"""
        print("\n[3] FREQUENCY BAND CORRELATION")
        print("-" * 70)

        # Define frequency bands
        bands = [
            ('Sub-Bass', 20, 80),
            ('Bass', 80, 250),
            ('Low-Mid', 250, 500),
            ('Mid', 500, 2000),
            ('High-Mid', 2000, 4000),
            ('Highs', 4000, 10000),
            ('Air', 10000, 20000)
        ]

        print(f"\n{'Band':<12} {'Range':<18} {'Correlation':<13} {'Interpretation'}")
        print("-" * 70)

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

            # Calculate correlation
            if np.std(left_filtered) > 1e-6 and np.std(right_filtered) > 1e-6:
                corr, _ = pearsonr(left_filtered, right_filtered)
            else:
                corr = 0.0

            # Interpretation
            if corr > 0.8:
                interpretation = "Narrow/Mono"
            elif corr > 0.5:
                interpretation = "Moderate"
            elif corr > 0.2:
                interpretation = "Wide"
            elif corr > 0:
                interpretation = "Very Wide"
            else:
                interpretation = "Phase Issue!"

            # Visual bar
            bar_pos = int((corr + 1) / 2 * 20)
            bar = '▓' * bar_pos + '░' * (20 - bar_pos)

            print(f"{band_name:<12} {low_freq:>5}-{high_freq:<8} Hz  {corr:+.3f} [{bar}] {interpretation}")

        print("\nObservations:")

        # Analyze patterns
        # Re-calculate for pattern detection
        bass_bands_corr = []
        mid_bands_corr = []
        high_bands_corr = []

        for band_name, low_freq, high_freq in bands:
            nyquist = self.sr / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.99)
            sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
            left_filtered = signal.sosfilt(sos, self.left)
            right_filtered = signal.sosfilt(sos, self.right)

            if np.std(left_filtered) > 1e-6 and np.std(right_filtered) > 1e-6:
                corr, _ = pearsonr(left_filtered, right_filtered)
            else:
                corr = 0.0

            if high_freq <= 500:
                bass_bands_corr.append(corr)
            elif high_freq <= 4000:
                mid_bands_corr.append(corr)
            else:
                high_bands_corr.append(corr)

        bass_avg = np.mean(bass_bands_corr)
        mid_avg = np.mean(mid_bands_corr)
        high_avg = np.mean(high_bands_corr)

        if bass_avg > 0.8:
            print("  • Bass is mostly MONO/CENTER (typical for low-end focus)")
        if mid_avg < 0.5:
            print("  • Mids have WIDE STEREO (good for spatial depth)")
        if high_avg < 0.3:
            print("  • Highs are VERY WIDE (creates air and space)")

        if bass_avg < mid_avg < high_avg:
            print("  • ⚠ INVERTED PATTERN: Wider bass than highs (unusual)")
        elif bass_avg > mid_avg > high_avg:
            print("  • TYPICAL PATTERN: Narrow bass, wider highs (standard mixing)")

    def calculate_phase_coherence(self):
        """Calculate phase coherence over time"""
        print("\n[4] PHASE COHERENCE ANALYSIS")
        print("-" * 70)

        # Use short-time windows to detect phase issues
        window_size = int(0.1 * self.sr)  # 100ms windows
        hop_size = window_size // 2

        correlations = []

        for i in range(0, len(self.left) - window_size, hop_size):
            left_window = self.left[i:i + window_size]
            right_window = self.right[i:i + window_size]

            if np.std(left_window) > 1e-6 and np.std(right_window) > 1e-6:
                corr, _ = pearsonr(left_window, right_window)
                correlations.append(corr)

        correlations = np.array(correlations)

        # Statistics
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)

        # Count phase problems
        phase_problem_count = np.sum(correlations < 0)
        phase_problem_percent = (phase_problem_count / len(correlations)) * 100

        print(f"\nMean Correlation: {mean_corr:+.4f}")
        print(f"Std Deviation:    {std_corr:.4f}")
        print(f"Range:            {min_corr:+.4f} to {max_corr:+.4f}")

        print(f"\nPhase Issues: {phase_problem_percent:.1f}% of time")

        if phase_problem_percent > 10:
            print("  ⚠ CRITICAL: Significant phase problems detected!")
            print("    This will cause serious issues in mono playback")
        elif phase_problem_percent > 5:
            print("  ⚠ WARNING: Some phase issues present")
            print("    May cause minor problems in mono")
        elif phase_problem_percent > 1:
            print("  ○ MINOR: Occasional phase issues (acceptable)")
        else:
            print("  ✓ GOOD: Minimal phase problems")

        # Stability check
        if std_corr < 0.1:
            print(f"\nStability: VERY STABLE (consistent stereo image)")
        elif std_corr < 0.2:
            print(f"\nStability: STABLE (typical variation)")
        else:
            print(f"\nStability: VARIABLE (dynamic stereo changes)")

    def check_mono_compatibility(self):
        """Check how the mix will sound in mono"""
        print("\n[5] MONO COMPATIBILITY CHECK")
        print("-" * 70)

        # Simulate mono mixdown
        mono = (self.left + self.right) / 2

        # Calculate RMS of stereo vs mono
        stereo_rms = np.sqrt((np.mean(self.left ** 2) + np.mean(self.right ** 2)) / 2)
        mono_rms = np.sqrt(np.mean(mono ** 2))

        # Level loss in dB
        if mono_rms > 0 and stereo_rms > 0:
            level_loss_db = 20 * np.log10(mono_rms / stereo_rms)
        else:
            level_loss_db = 0

        print(f"\nStereo RMS: {stereo_rms:.6f}")
        print(f"Mono RMS:   {mono_rms:.6f}")
        print(f"\nLevel Loss in Mono: {level_loss_db:+.2f} dB")

        # Interpretation
        if level_loss_db > -0.5:
            print("\n✓ EXCELLENT mono compatibility")
            print("  Minimal level loss, no phase cancellation")
        elif level_loss_db > -1.5:
            print("\n✓ GOOD mono compatibility")
            print("  Slight level loss, acceptable for most uses")
        elif level_loss_db > -3.0:
            print("\n○ FAIR mono compatibility")
            print("  Noticeable level loss, some phase issues")
        else:
            print("\n✗ POOR mono compatibility")
            print("  Significant level loss due to phase cancellation")
            print("  ⚠ Will sound thin/weak on mono systems")

        # Calculate correlation of mono version
        mono_corr, _ = pearsonr(self.left, self.right)

        if mono_corr < 0:
            print("\n⚠ CRITICAL WARNING: Negative correlation detected!")
            print("  Out-of-phase content will CANCEL in mono")
            print("  Action needed: Check phase relationships in mix")

    def generate_summary(self):
        """Generate overall summary"""
        print("\n" + "=" * 70)
        print("STEREO FIELD SUMMARY")
        print("=" * 70)

        # Overall correlation
        correlation, _ = pearsonr(self.left, self.right)

        # Mid/Side
        mid = (self.left + self.right) / 2
        side = (self.left - self.right) / 2
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))

        if mid_rms + side_rms > 0:
            stereo_width = (2 * side_rms) / (mid_rms + side_rms)
        else:
            stereo_width = 0

        # Mono compatibility
        mono = (self.left + self.right) / 2
        stereo_rms = np.sqrt((np.mean(self.left ** 2) + np.mean(self.right ** 2)) / 2)
        mono_rms = np.sqrt(np.mean(mono ** 2))

        if mono_rms > 0 and stereo_rms > 0:
            level_loss_db = 20 * np.log10(mono_rms / stereo_rms)
        else:
            level_loss_db = 0

        print(f"\nKey Metrics:")
        print(f"  • Stereo Correlation:  {correlation:+.3f}")
        print(f"  • Stereo Width:        {stereo_width:.3f}")
        print(f"  • Mono Level Loss:     {level_loss_db:+.2f} dB")

        print(f"\nOverall Assessment:")

        # Determine mix style
        if correlation > 0.8 and stereo_width < 0.3:
            print("  → MONO/CENTER-FOCUSED MIX")
            print("    Limited stereo information, mono-compatible but narrow")
        elif correlation > 0.5 and 0.3 <= stereo_width <= 0.6:
            print("  → BALANCED STEREO MIX")
            print("    Good balance of center and stereo content")
        elif correlation < 0.5 and stereo_width > 0.6:
            print("  → WIDE STEREO MIX")
            print("    Significant stereo separation, check mono compatibility")

        if level_loss_db < -3:
            print("  ⚠ MONO COMPATIBILITY CONCERNS")
        else:
            print("  ✓ MONO COMPATIBLE")

        if correlation < 0:
            print("  ⚠⚠⚠ PHASE ISSUES DETECTED - REQUIRES ATTENTION")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "trimmed_output33.wav"  # Change this to your WAV file name

    print("\n" + "=" * 70)
    print("STEREO CORRELATION ANALYZER")
    print("=" * 70)
    print("\nAnalyzing phase and level relationship between L/R channels")
    print("Range: +1 (mono/identical) to -1 (phase problems)")
    print()

    # Initialize and run
    analyzer = StereoCorrelationAnalyzer(AUDIO_FILE)
    analyzer.load_audio()
    analyzer.calculate_correlation()
    analyzer.calculate_mid_side_analysis()
    analyzer.calculate_frequency_band_correlation()
    analyzer.calculate_phase_coherence()
    analyzer.check_mono_compatibility()
    analyzer.generate_summary()