# Drum heaviness refers to a sound that feels powerful, full, and impactful, achieved through a
# combination of low-end frequencies, controlled resonance, strong attack (transients), and purposeful decay

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
from scipy import signal

warnings.filterwarnings('ignore')


class DrumHeavinessAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate drums"""
        print("=" * 70)
        print("DRUM HEAVINESS ANALYZER")
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
        print("Separating drums...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract drums (index 0)
        self.stems['drums'] = librosa.to_mono(sources[0].numpy())

        print("✓ Separation complete!\n")

    def calculate_drum_heaviness(self):
        """Analyze drum heaviness (power, impact, low-end)"""
        print("=" * 70)
        print("DRUM HEAVINESS ANALYSIS")
        print("=" * 70)

        drums = self.stems['drums']

        # Check if drums are present
        drums_rms = np.sqrt(np.mean(drums ** 2))
        if drums_rms < 1e-6:
            print("\n⚠ No significant drum content detected!")
            print("This may be a track without percussion.\n")
            return

        # 1. LOW-FREQUENCY CONTENT (kick drum power)
        print("\n[1] LOW-FREQUENCY POWER (Kick Drum Weight)")
        print("-" * 70)

        stft = librosa.stft(drums, n_fft=4096)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Sub-bass (20-60 Hz) - deep kick
        sub_mask = (freqs >= 20) & (freqs < 60)
        sub_energy = np.sum(magnitude[sub_mask, :])

        # Kick fundamentals (60-150 Hz)
        kick_mask = (freqs >= 60) & (freqs < 150)
        kick_energy = np.sum(magnitude[kick_mask, :])

        # Snare body (150-500 Hz)
        snare_mask = (freqs >= 150) & (freqs < 500)
        snare_energy = np.sum(magnitude[snare_mask, :])

        # High-mids (500-2000 Hz) - attack/snap
        mid_mask = (freqs >= 500) & (freqs < 2000)
        mid_energy = np.sum(magnitude[mid_mask, :])

        # Highs (2000-10000 Hz) - cymbals/hi-hats
        high_mask = (freqs >= 2000) & (freqs < 10000)
        high_energy = np.sum(magnitude[high_mask, :])

        total_energy = sub_energy + kick_energy + snare_energy + mid_energy + high_energy

        if total_energy > 0:
            sub_percent = (sub_energy / total_energy) * 100
            kick_percent = (kick_energy / total_energy) * 100
            snare_percent = (snare_energy / total_energy) * 100
            mid_percent = (mid_energy / total_energy) * 100
            high_percent = (high_energy / total_energy) * 100
        else:
            sub_percent = kick_percent = snare_percent = mid_percent = high_percent = 0

        low_end_total = sub_percent + kick_percent

        print(f"\nSub-Bass (20-60 Hz):   {sub_percent:>6.2f}%  {'█' * int(sub_percent / 2)}")
        print(f"Kick (60-150 Hz):      {kick_percent:>6.2f}%  {'█' * int(kick_percent / 2)}")
        print(f"Snare (150-500 Hz):    {snare_percent:>6.2f}%  {'█' * int(snare_percent / 2)}")
        print(f"Attack (500-2000 Hz):  {mid_percent:>6.2f}%  {'█' * int(mid_percent / 2)}")
        print(f"Cymbals (2-10 kHz):    {high_percent:>6.2f}%  {'█' * int(high_percent / 2)}")

        print(f"\nTotal Low-End (20-150 Hz): {low_end_total:.1f}%")

        # 2. TRANSIENT STRENGTH (attack/impact)
        print("\n[2] TRANSIENT STRENGTH (Attack/Impact)")
        print("-" * 70)

        # Detect onsets
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=drums, sr=self.sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True
        )

        # Analyze transient strength
        onset_strength_mean = np.mean(onset_env)
        onset_strength_max = np.max(onset_env)

        print(f"\nOnset Strength (mean): {onset_strength_mean:.3f}")
        print(f"Onset Strength (max):  {onset_strength_max:.3f}")
        print(f"Number of hits:        {len(onsets)}")

        if onset_strength_mean > 10:
            print("  → VERY STRONG transients (powerful, impactful)")
        elif onset_strength_mean > 5:
            print("  → STRONG transients (punchy)")
        elif onset_strength_mean > 2:
            print("  → MODERATE transients (typical)")
        else:
            print("  → WEAK transients (soft, subtle)")

        # 3. RMS ENERGY AND COMPRESSION
        print("\n[3] POWER AND DENSITY")
        print("-" * 70)

        rms = librosa.feature.rms(y=drums)[0]
        rms_mean = np.mean(rms)
        rms_db = 20 * np.log10(rms_mean + 1e-10)

        print(f"\nRMS Energy: {rms_mean:.6f} ({rms_db:.1f} dB)")

        # Crest factor (peak-to-average ratio)
        peaks = np.abs(drums)
        peak_value = np.max(peaks)
        if rms_mean > 0:
            crest_factor = peak_value / rms_mean
            crest_db = 20 * np.log10(crest_factor)

            print(f"Crest Factor: {crest_factor:.2f} ({crest_db:.1f} dB)")

            if crest_db < 10:
                print("  → HEAVILY COMPRESSED (dense, sustained)")
            elif crest_db < 15:
                print("  → MODERATELY COMPRESSED (controlled)")
            elif crest_db < 20:
                print("  → NATURAL DYNAMICS (typical)")
            else:
                print("  → VERY DYNAMIC (loose, expressive)")

        # 4. SUSTAIN AND DECAY
        print("\n[4] SUSTAIN AND RESONANCE")
        print("-" * 70)

        # Analyze envelope decay
        envelope = np.abs(signal.hilbert(drums))

        # Find peaks and measure decay times
        from scipy.signal import find_peaks
        peaks_idx, properties = find_peaks(envelope, height=np.max(envelope) * 0.3, distance=int(0.1 * self.sr))

        decay_times = []
        for peak_idx in peaks_idx[:min(20, len(peaks_idx))]:  # Analyze first 20 hits
            # Measure time to decay to 40% of peak
            peak_value = envelope[peak_idx]
            threshold = peak_value * 0.4

            decay_idx = peak_idx
            for i in range(peak_idx, min(peak_idx + int(0.5 * self.sr), len(envelope))):
                if envelope[i] < threshold:
                    decay_idx = i
                    break

            decay_time = (decay_idx - peak_idx) / self.sr
            decay_times.append(decay_time)

        if decay_times:
            avg_decay = np.mean(decay_times)
            print(f"\nAverage Decay Time: {avg_decay * 1000:.1f} ms")

            if avg_decay > 0.3:
                print("  → LONG DECAY (resonant, open, ringing)")
            elif avg_decay > 0.15:
                print("  → MODERATE DECAY (natural)")
            elif avg_decay > 0.08:
                print("  → SHORT DECAY (tight, controlled)")
            else:
                print("  → VERY SHORT DECAY (gated, clipped)")

        # 5. SPECTRAL CENTROID (tonality)
        print("\n[5] TONAL CHARACTER")
        print("-" * 70)

        centroid = librosa.feature.spectral_centroid(y=drums, sr=self.sr)
        centroid_mean = np.mean(centroid)

        print(f"\nSpectral Centroid: {centroid_mean:.1f} Hz")

        if centroid_mean < 500:
            print("  → VERY DARK/LOW (bass-heavy, boomy)")
        elif centroid_mean < 1000:
            print("  → DARK (warm, full-bodied)")
        elif centroid_mean < 2000:
            print("  → BALANCED (even tonal distribution)")
        elif centroid_mean < 3500:
            print("  → BRIGHT (crisp, clear)")
        else:
            print("  → VERY BRIGHT (cymbal-heavy, splashy)")

        # 6. CALCULATE COMPOSITE HEAVINESS SCORE
        print("\n[6] COMPOSITE HEAVINESS SCORE")
        print("-" * 70)

        # Factors contributing to heaviness:
        # 1. Low-frequency content (higher = heavier)
        low_factor = min(low_end_total / 40.0, 1.0)

        # 2. High onset strength (stronger = heavier)
        onset_factor = min(onset_strength_mean / 15.0, 1.0)

        # 3. High RMS energy (louder = heavier)
        energy_factor = min((rms_db + 40) / 40.0, 1.0)
        energy_factor = max(0.0, energy_factor)

        # 4. Low crest factor = compressed = heavier/denser
        if rms_mean > 0:
            compression_factor = max(0, 1.0 - (crest_db - 10) / 15)
            compression_factor = min(1.0, max(0.0, compression_factor))
        else:
            compression_factor = 0.5

        # 5. Long decay = resonant = heavier feel
        if decay_times:
            avg_decay = np.mean(decay_times)
            sustain_factor = min(avg_decay / 0.4, 1.0)
        else:
            sustain_factor = 0.5

        # 6. Low spectral centroid = darker = heavier
        centroid_factor = max(0, 1.0 - (centroid_mean - 300) / 2500)
        centroid_factor = min(1.0, max(0.0, centroid_factor))

        # Weighted combination
        heaviness_score = (
                0.30 * low_factor +  # Most important
                0.25 * onset_factor +
                0.15 * energy_factor +
                0.15 * compression_factor +
                0.10 * sustain_factor +
                0.05 * centroid_factor
        )

        heaviness_score = min(1.0, max(0.0, heaviness_score))

        print(f"\nComponent Scores:")
        print(f"  Low-Frequency Content:  {low_factor:.3f}")
        print(f"  Transient Strength:     {onset_factor:.3f}")
        print(f"  Energy Level:           {energy_factor:.3f}")
        print(f"  Compression:            {compression_factor:.3f}")
        print(f"  Sustain/Resonance:      {sustain_factor:.3f}")
        print(f"  Dark Tone:              {centroid_factor:.3f}")

        print(f"\n{'=' * 35}")
        print(f"DRUM HEAVINESS SCORE: {heaviness_score:.3f}")
        print(f"{'=' * 35}")

        # Visual bar
        heaviness_percent = int(heaviness_score * 100)
        bar = '█' * heaviness_percent + '░' * (100 - heaviness_percent)
        print(f"\n[{bar}]")
        print(" Light                                                     Heavy")

        # Rating
        if heaviness_score < 0.2:
            rating = "VERY LIGHT"
            description = "Subtle, delicate percussion; hi-hat/cymbal focused"
        elif heaviness_score < 0.4:
            rating = "LIGHT"
            description = "Moderate impact with some low-end presence"
        elif heaviness_score < 0.6:
            rating = "MODERATE"
            description = "Balanced drum weight, typical rock/pop kit"
        elif heaviness_score < 0.8:
            rating = "HEAVY"
            description = "Powerful, impactful drums with strong low-end"
        else:
            rating = "VERY HEAVY"
            description = "Massive, crushing drum sound; metal/EDM style"

        print(f"\nRating: {rating}")
        print(f"{description}")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = DrumHeavinessAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_drum_heaviness()