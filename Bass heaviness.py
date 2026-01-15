# Bass heaviness in sound refers to a prominent, deep, and often powerful low-frequency presence,
# felt physically as much as heard, achieved through specific music genres (like Hip-Hop, EDM, Dubstep),
# audio gear (powerful amps, subwoofers), sound design (distorted, textured bass sounds), and acoustics,
# creating a sense of "weight" or impact beyond simple loudness.

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
from scipy import signal

warnings.filterwarnings('ignore')


class BassHeavinessAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.stems = {}

    def separate_stems(self):
        """Use Demucs to separate bass"""
        print("=" * 70)
        print("BASS HEAVINESS ANALYZER")
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
        print("Separating bass...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract bass (index 1)
        self.stems['bass'] = librosa.to_mono(sources[1].numpy())

        print("✓ Separation complete!\n")

    def calculate_bass_heaviness(self):
        """Analyze bass heaviness (depth, power, physical impact)"""
        print("=" * 70)
        print("BASS HEAVINESS ANALYSIS")
        print("=" * 70)

        bass = self.stems['bass']

        # Check if bass is present
        bass_rms = np.sqrt(np.mean(bass ** 2))
        if bass_rms < 1e-6:
            print("\n⚠ No significant bass content detected!")
            print("This may be a track without bass instruments.\n")
            return

        # 1. SUB-BASS CONTENT (physical impact)
        print("\n[1] SUB-BASS CONTENT (Physical Impact)")
        print("-" * 70)

        stft = librosa.stft(bass, n_fft=4096)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Sub-bass (20-60 Hz) - "feel it in your chest"
        sub_mask = (freqs >= 20) & (freqs < 60)
        sub_energy = np.sum(magnitude[sub_mask, :])

        # Deep bass (60-80 Hz)
        deep_mask = (freqs >= 60) & (freqs < 80)
        deep_energy = np.sum(magnitude[deep_mask, :])

        # Low bass (80-120 Hz) - fundamental bass notes
        low_mask = (freqs >= 80) & (freqs < 120)
        low_energy = np.sum(magnitude[low_mask, :])

        # Mid-bass (120-250 Hz) - clarity/definition
        mid_mask = (freqs >= 120) & (freqs < 250)
        mid_energy = np.sum(magnitude[mid_mask, :])

        # Upper bass (250-500 Hz) - presence/attack
        upper_mask = (freqs >= 250) & (freqs < 500)
        upper_energy = np.sum(magnitude[upper_mask, :])

        total_energy = sub_energy + deep_energy + low_energy + mid_energy + upper_energy

        if total_energy > 0:
            sub_percent = (sub_energy / total_energy) * 100
            deep_percent = (deep_energy / total_energy) * 100
            low_percent = (low_energy / total_energy) * 100
            mid_percent = (mid_energy / total_energy) * 100
            upper_percent = (upper_energy / total_energy) * 100
        else:
            sub_percent = deep_percent = low_percent = mid_percent = upper_percent = 0

        deep_bass_total = sub_percent + deep_percent

        print(f"\nSub-Bass (20-60 Hz):   {sub_percent:>6.2f}%  {'█' * int(sub_percent / 2)}")
        print(f"Deep (60-80 Hz):       {deep_percent:>6.2f}%  {'█' * int(deep_percent / 2)}")
        print(f"Low (80-120 Hz):       {low_percent:>6.2f}%  {'█' * int(low_percent / 2)}")
        print(f"Mid (120-250 Hz):      {mid_percent:>6.2f}%  {'█' * int(mid_percent / 2)}")
        print(f"Upper (250-500 Hz):    {upper_percent:>6.2f}%  {'█' * int(upper_percent / 2)}")

        print(f"\nDeep Bass Content (20-80 Hz): {deep_bass_total:.1f}%")

        if deep_bass_total > 40:
            print("  → EXTREMELY DEEP (club/EDM style, subwoofer essential)")
        elif deep_bass_total > 25:
            print("  → VERY DEEP (strong sub-bass presence)")
        elif deep_bass_total > 15:
            print("  → MODERATELY DEEP (standard modern production)")
        else:
            print("  → LIGHT SUB-BASS (traditional/acoustic bass)")

        # 2. RMS ENERGY AND LOUDNESS
        print("\n[2] POWER AND LOUDNESS")
        print("-" * 70)

        rms = librosa.feature.rms(y=bass)[0]
        rms_mean = np.mean(rms)
        rms_db = 20 * np.log10(rms_mean + 1e-10)

        print(f"\nRMS Energy: {rms_mean:.6f} ({rms_db:.1f} dB)")

        if rms_db > -10:
            print("  → VERY LOUD bass (dominant in mix)")
        elif rms_db > -20:
            print("  → LOUD bass (prominent)")
        elif rms_db > -30:
            print("  → MODERATE level (balanced)")
        else:
            print("  → QUIET bass (subtle)")

        # Crest factor (sustain vs attack)
        peaks = np.abs(bass)
        peak_value = np.max(peaks)
        if rms_mean > 0:
            crest_factor = peak_value / rms_mean
            crest_db = 20 * np.log10(crest_factor)

            print(f"Crest Factor: {crest_factor:.2f} ({crest_db:.1f} dB)")

            if crest_db < 8:
                print("  → HEAVILY COMPRESSED (sustained, dense)")
            elif crest_db < 14:
                print("  → MODERATELY COMPRESSED (controlled)")
            elif crest_db < 20:
                print("  → NATURAL DYNAMICS (typical)")
            else:
                print("  → VERY DYNAMIC (fingerstyle/slap bass)")

        # 3. DISTORTION/HARMONICS (texture)
        print("\n[3] HARMONIC CONTENT (Texture/Distortion)")
        print("-" * 70)

        # Analyze harmonic content above fundamental
        # Higher harmonics = more distortion/grit
        harmonic_mask = (freqs >= 500) & (freqs < 2000)
        harmonic_energy = np.sum(magnitude[harmonic_mask, :])

        fundamental_mask = (freqs >= 60) & (freqs < 250)
        fundamental_energy = np.sum(magnitude[fundamental_mask, :])

        if fundamental_energy > 0:
            harmonic_ratio = harmonic_energy / fundamental_energy
            print(f"\nHarmonic-to-Fundamental Ratio: {harmonic_ratio:.3f}")

            if harmonic_ratio > 0.5:
                print("  → HIGH harmonics (distorted, gritty, saturated)")
            elif harmonic_ratio > 0.2:
                print("  → MODERATE harmonics (textured, warm)")
            else:
                print("  → CLEAN bass (pure, sub-focused)")

        # Spectral flatness (noisy vs tonal)
        flatness = librosa.feature.spectral_flatness(y=bass)
        flatness_mean = np.mean(flatness)

        print(f"Spectral Flatness: {flatness_mean:.3f}")

        if flatness_mean > 0.3:
            print("  → NOISY/FUZZY (heavily distorted)")
        elif flatness_mean > 0.15:
            print("  → TEXTURED (moderate distortion)")
        else:
            print("  → TONAL (clean, pure)")

        # 4. ATTACK AND SUSTAIN
        print("\n[4] ATTACK AND SUSTAIN")
        print("-" * 70)

        # Analyze envelope
        envelope = np.abs(signal.hilbert(bass))

        # Find attacks
        from scipy.signal import find_peaks
        peaks_idx, properties = find_peaks(envelope, height=np.max(envelope) * 0.3, distance=int(0.1 * self.sr))

        if len(peaks_idx) > 0:
            # Measure attack time (first 10% to 90% of peak)
            attack_times = []
            sustain_times = []

            for peak_idx in peaks_idx[:min(20, len(peaks_idx))]:
                peak_value = envelope[peak_idx]

                # Attack: find 10% point before peak
                start_threshold = peak_value * 0.1
                attack_start = peak_idx
                for i in range(peak_idx, max(0, peak_idx - int(0.1 * self.sr)), -1):
                    if envelope[i] < start_threshold:
                        attack_start = i
                        break

                attack_time = (peak_idx - attack_start) / self.sr
                attack_times.append(attack_time)

                # Sustain: time to decay to 50% of peak
                sustain_threshold = peak_value * 0.5
                sustain_end = peak_idx
                for i in range(peak_idx, min(len(envelope), peak_idx + int(0.5 * self.sr))):
                    if envelope[i] < sustain_threshold:
                        sustain_end = i
                        break

                sustain_time = (sustain_end - peak_idx) / self.sr
                sustain_times.append(sustain_time)

            if attack_times:
                avg_attack = np.mean(attack_times)
                print(f"\nAverage Attack Time: {avg_attack * 1000:.1f} ms")

                if avg_attack < 0.01:
                    print("  → VERY FAST attack (slap/pick bass, punchy)")
                elif avg_attack < 0.03:
                    print("  → FAST attack (typical electric bass)")
                elif avg_attack < 0.05:
                    print("  → MODERATE attack (fingerstyle)")
                else:
                    print("  → SLOW attack (bowed/synth bass)")

            if sustain_times:
                avg_sustain = np.mean(sustain_times)
                print(f"Average Sustain Time: {avg_sustain * 1000:.1f} ms")

                if avg_sustain > 0.3:
                    print("  → LONG sustain (held notes, pads)")
                elif avg_sustain > 0.15:
                    print("  → MODERATE sustain (typical)")
                elif avg_sustain > 0.08:
                    print("  → SHORT sustain (staccato)")
                else:
                    print("  → VERY SHORT sustain (muted/damped)")

        # 5. SPECTRAL CENTROID (tonality)
        print("\n[5] TONAL CHARACTER")
        print("-" * 70)

        centroid = librosa.feature.spectral_centroid(y=bass, sr=self.sr)
        centroid_mean = np.mean(centroid)

        print(f"\nSpectral Centroid: {centroid_mean:.1f} Hz")

        if centroid_mean < 100:
            print("  → ULTRA-DEEP (sub-focused, felt more than heard)")
        elif centroid_mean < 150:
            print("  → VERY DEEP (dark, boomy)")
        elif centroid_mean < 250:
            print("  → DEEP (warm, full)")
        elif centroid_mean < 400:
            print("  → BALANCED (clear definition)")
        else:
            print("  → BRIGHT (midrange-heavy, defined)")

        # 6. CALCULATE COMPOSITE HEAVINESS SCORE
        print("\n[6] COMPOSITE HEAVINESS SCORE")
        print("-" * 70)

        # Factors contributing to heaviness:
        # 1. Sub-bass content (higher = heavier/more physical)
        sub_factor = min(deep_bass_total / 35.0, 1.0)

        # 2. High RMS energy (louder = heavier)
        energy_factor = min((rms_db + 35) / 35.0, 1.0)
        energy_factor = max(0.0, energy_factor)

        # 3. Low crest factor = compressed/sustained = heavier
        if rms_mean > 0:
            compression_factor = max(0, 1.0 - (crest_db - 8) / 16)
            compression_factor = min(1.0, max(0.0, compression_factor))
        else:
            compression_factor = 0.5

        # 4. Harmonic content (distortion = heavier/more aggressive)
        if fundamental_energy > 0:
            harmonic_ratio = harmonic_energy / fundamental_energy
            distortion_factor = min(harmonic_ratio / 0.6, 1.0)
        else:
            distortion_factor = 0.5

        # 5. Long sustain = more weight
        if sustain_times:
            avg_sustain = np.mean(sustain_times)
            sustain_factor = min(avg_sustain / 0.4, 1.0)
        else:
            sustain_factor = 0.5

        # 6. Low spectral centroid = darker/heavier
        centroid_factor = max(0, 1.0 - (centroid_mean - 80) / 300)
        centroid_factor = min(1.0, max(0.0, centroid_factor))

        # Weighted combination
        heaviness_score = (
                0.35 * sub_factor +  # Most important
                0.25 * energy_factor +
                0.15 * compression_factor +
                0.10 * distortion_factor +
                0.10 * sustain_factor +
                0.05 * centroid_factor
        )

        heaviness_score = min(1.0, max(0.0, heaviness_score))

        print(f"\nComponent Scores:")
        print(f"  Sub-Bass Content:       {sub_factor:.3f}")
        print(f"  Energy Level:           {energy_factor:.3f}")
        print(f"  Compression/Sustain:    {compression_factor:.3f}")
        print(f"  Harmonic Distortion:    {distortion_factor:.3f}")
        print(f"  Note Sustain:           {sustain_factor:.3f}")
        print(f"  Deep Tone:              {centroid_factor:.3f}")

        print(f"\n{'=' * 35}")
        print(f"BASS HEAVINESS SCORE: {heaviness_score:.3f}")
        print(f"{'=' * 35}")

        # Visual bar
        heaviness_percent = int(heaviness_score * 100)
        bar = '█' * heaviness_percent + '░' * (100 - heaviness_percent)
        print(f"\n[{bar}]")
        print(" Light                                                     Heavy")

        # Rating
        if heaviness_score < 0.2:
            rating = "VERY LIGHT"
            description = "Subtle, acoustic-style bass; minimal low-end impact"
        elif heaviness_score < 0.4:
            rating = "LIGHT"
            description = "Moderate bass presence, clear but not overwhelming"
        elif heaviness_score < 0.6:
            rating = "MODERATE"
            description = "Solid bass foundation, typical modern production"
        elif heaviness_score < 0.8:
            rating = "HEAVY"
            description = "Powerful, physical bass with strong sub presence"
        else:
            rating = "VERY HEAVY"
            description = "Massive, crushing bass; club/EDM/dubstep style"

        print(f"\nRating: {rating}")
        print(f"{description}")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    AUDIO_FILE = "trimmed_output33.wav"  # Change this

    analyzer = BassHeavinessAnalyzer(AUDIO_FILE)
    analyzer.separate_stems()
    analyzer.calculate_bass_heaviness()