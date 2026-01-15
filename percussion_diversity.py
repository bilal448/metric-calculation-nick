# Percussion instruments offer immense sound diversity by creating sounds through striking, shaking, or
# scraping, ranging from definite pitches (marimba, timpani) to indefinite ones (cymbals, snare drums)

import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats, signal

warnings.filterwarnings('ignore')


class PercussionDiversityAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sr = 44100
        self.drums = None
        self.onset_segments = []
        self.features_matrix = None
        self.results = {}

    def separate_drums(self):
        """Use Demucs to separate drum track"""
        print(f"Loading audio: {self.audio_file}")

        # Load audio
        audio, sr = librosa.load(self.audio_file, sr=self.sr, mono=False)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        # Load Demucs model
        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.eval()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Separate sources
        print("Separating drums...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')[0]

        # Extract drums (index 1 in htdemucs: drums, bass, other, vocals)
        drums_stereo = sources[1].numpy()

        # Convert to mono for analysis
        self.drums = librosa.to_mono(drums_stereo)
        print("Drum separation complete!")

    def extract_onset_segments(self, segment_duration=0.3):
        """Extract individual drum hit segments for analysis"""
        print("\nExtracting individual drum hit segments...")

        hop_length = 512

        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=self.drums, sr=self.sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            backtrack=True
        )

        onset_samples = librosa.frames_to_samples(onsets, hop_length=hop_length)

        # Extract segments around each onset
        segment_length = int(segment_duration * self.sr)

        for onset in onset_samples:
            start = onset
            end = min(onset + segment_length, len(self.drums))

            if end - start > self.sr * 0.05:  # At least 50ms
                segment = self.drums[start:end]
                self.onset_segments.append(segment)

        print(f"   Extracted {len(self.onset_segments)} drum hit segments")
        self.results['num_segments'] = len(self.onset_segments)

    def compute_spectral_features(self, segment):
        """Compute spectral features for a single segment"""
        features = {}

        # Compute STFT
        stft = librosa.stft(segment)
        magnitude = np.abs(stft)

        # 1. Spectral Centroid - brightness
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)
        features['spectral_centroid'] = np.mean(centroid)

        # 2. Spectral Spread (Variance) - spread around centroid
        spread = np.sqrt(np.sum(((librosa.fft_frequencies(sr=self.sr, n_fft=2048)[:magnitude.shape[0]] - features[
            'spectral_centroid']) ** 2) * magnitude.mean(axis=1)) / np.sum(magnitude.mean(axis=1)))
        features['spectral_spread'] = spread

        # 3. Spectral Roll-off - frequency below which 85% of energy is contained
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sr, roll_percent=0.85)
        features['spectral_rolloff'] = np.mean(rolloff)

        # 4. Spectral Flatness - measure of noisiness
        flatness = librosa.feature.spectral_flatness(S=magnitude)
        features['spectral_flatness'] = np.mean(flatness)

        # 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sr)
        features['spectral_bandwidth'] = np.mean(bandwidth)

        # 6. Zero Crossing Rate - timbral texture
        zcr = librosa.feature.zero_crossing_rate(segment)
        features['zero_crossing_rate'] = np.mean(zcr)

        # 7. Odd-to-Even Harmonic Energy Ratio
        # Estimate fundamental frequency
        f0 = librosa.yin(segment, fmin=50, fmax=2000, sr=self.sr)
        f0_mean = np.nanmean(f0)

        if not np.isnan(f0_mean) and f0_mean > 0:
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

            # Calculate odd and even harmonic energies
            odd_energy = 0
            even_energy = 0

            for harmonic in range(1, 10):
                target_freq = f0_mean * harmonic
                if target_freq < self.sr / 2:
                    idx = np.argmin(np.abs(freqs - target_freq))
                    energy = np.mean(magnitude[idx, :])

                    if harmonic % 2 == 1:  # Odd harmonic
                        odd_energy += energy
                    else:  # Even harmonic
                        even_energy += energy

            if even_energy > 0:
                features['odd_even_ratio'] = odd_energy / even_energy
            else:
                features['odd_even_ratio'] = odd_energy
        else:
            features['odd_even_ratio'] = 0

        # 8. Inharmonicity - deviation from perfect harmonic series
        if not np.isnan(f0_mean) and f0_mean > 0:
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
            spectral_peaks_idx = librosa.util.peak_pick(magnitude.mean(axis=1), pre_max=3, post_max=3, pre_avg=3,
                                                        post_avg=3, delta=0.1, wait=10)
            spectral_peaks = freqs[spectral_peaks_idx]

            inharmonicity_sum = 0
            harmonic_count = 0

            for harmonic in range(2, min(8, len(spectral_peaks))):
                expected_freq = f0_mean * harmonic
                # Find closest peak
                if len(spectral_peaks) > harmonic:
                    actual_freq = spectral_peaks[harmonic]
                    deviation = abs(actual_freq - expected_freq) / expected_freq
                    inharmonicity_sum += deviation
                    harmonic_count += 1

            if harmonic_count > 0:
                features['inharmonicity'] = inharmonicity_sum / harmonic_count
            else:
                features['inharmonicity'] = 0
        else:
            features['inharmonicity'] = 0

        return features

    def compute_temporal_features(self, segment):
        """Compute temporal features for a single segment"""
        features = {}

        # Envelope
        envelope = np.abs(signal.hilbert(segment))

        # 1. Attack Time - time to reach maximum amplitude
        max_idx = np.argmax(envelope)
        attack_time = max_idx / self.sr
        features['attack_time'] = attack_time

        # 2. Decay Time - time to decay to 40% of maximum after attack
        max_amp = envelope[max_idx]
        threshold = max_amp * 0.4

        decay_idx = max_idx
        for i in range(max_idx, len(envelope)):
            if envelope[i] < threshold:
                decay_idx = i
                break

        decay_time = (decay_idx - max_idx) / self.sr
        features['decay_time'] = decay_time

        # 3. RMS Energy
        rms = librosa.feature.rms(y=segment)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # 4. Temporal Centroid - center of energy in time
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)
        features['temporal_centroid'] = temporal_centroid / self.sr

        # 5. Log Attack Time - better perceptual scale
        features['log_attack_time'] = np.log10(attack_time + 1e-6)

        return features

    def extract_all_features(self):
        """Extract all features from all segments"""
        print("\nExtracting spectral and temporal features...")

        all_features = []
        feature_names = None

        for i, segment in enumerate(self.onset_segments):
            if (i + 1) % 10 == 0:
                print(f"   Processing segment {i + 1}/{len(self.onset_segments)}...")

            # Extract features
            spectral = self.compute_spectral_features(segment)
            temporal = self.compute_temporal_features(segment)

            # Combine all features
            combined = {**spectral, **temporal}

            if feature_names is None:
                feature_names = list(combined.keys())

            all_features.append([combined[key] for key in feature_names])

        # Convert to numpy array
        self.features_matrix = np.array(all_features)
        self.feature_names = feature_names

        print(f"   Extracted {len(feature_names)} features from {len(self.onset_segments)} segments")

        # Store individual feature statistics
        self.results['features'] = {}
        for i, name in enumerate(feature_names):
            self.results['features'][name] = {
                'mean': float(np.mean(self.features_matrix[:, i])),
                'std': float(np.std(self.features_matrix[:, i])),
                'min': float(np.min(self.features_matrix[:, i])),
                'max': float(np.max(self.features_matrix[:, i]))
            }

    def calculate_diversity_metrics(self):
        """Calculate diversity metrics from feature distributions"""
        print("\n" + "=" * 60)
        print("CALCULATING PERCUSSION SOUND DIVERSITY")
        print("=" * 60)

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.features_matrix)

        # 1. Overall Feature Variance (Diversity)
        print("\n[1] Feature-wise Diversity (Variance)...")
        feature_variances = np.var(features_normalized, axis=0)
        overall_variance = np.mean(feature_variances)

        self.results['diversity_metrics'] = {}
        self.results['diversity_metrics']['overall_variance'] = float(overall_variance)
        self.results['diversity_metrics']['feature_variances'] = {
            name: float(var) for name, var in zip(self.feature_names, feature_variances)
        }

        print(f"   Overall diversity score: {overall_variance:.3f}")
        print(f"   (Higher = more diverse percussion sounds)")

        # 2. Principal Component Analysis - dimensionality of timbre space
        print("\n[2] Timbre Space Dimensionality (PCA)...")
        pca = PCA()
        pca.fit(features_normalized)

        # Calculate effective dimensionality (number of components for 95% variance)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1

        self.results['diversity_metrics']['pca_components_95'] = int(n_components_95)
        self.results['diversity_metrics']['pca_variance_ratios'] = pca.explained_variance_ratio_.tolist()

        print(f"   Effective dimensions: {n_components_95}")
        print(f"   (More dimensions = more diverse timbral characteristics)")
        print(f"   First 3 PCs explain: {cumsum[2] * 100:.1f}% of variance")

        # 3. Pairwise Distance Metrics
        print("\n[3] Pairwise Timbre Distances...")
        from scipy.spatial.distance import pdist

        # Calculate pairwise Euclidean distances
        distances = pdist(features_normalized, metric='euclidean')

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        max_distance = np.max(distances)

        self.results['diversity_metrics']['mean_pairwise_distance'] = float(mean_distance)
        self.results['diversity_metrics']['std_pairwise_distance'] = float(std_distance)
        self.results['diversity_metrics']['max_pairwise_distance'] = float(max_distance)

        print(f"   Mean distance: {mean_distance:.3f}")
        print(f"   Std distance: {std_distance:.3f}")
        print(f"   Max distance: {max_distance:.3f}")
        print(f"   (Higher = more diverse/dissimilar sounds)")

        # 4. Spectral Diversity Score
        print("\n[4] Spectral Diversity Score...")
        spectral_features = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff',
                             'spectral_flatness', 'spectral_bandwidth']
        spectral_indices = [i for i, name in enumerate(self.feature_names) if name in spectral_features]

        if spectral_indices:
            spectral_data = features_normalized[:, spectral_indices]
            spectral_variance = np.mean(np.var(spectral_data, axis=0))
            self.results['diversity_metrics']['spectral_diversity'] = float(spectral_variance)
            print(f"   Spectral diversity: {spectral_variance:.3f}")

        # 5. Temporal Diversity Score
        print("\n[5] Temporal Diversity Score...")
        temporal_features = ['attack_time', 'decay_time', 'temporal_centroid']
        temporal_indices = [i for i, name in enumerate(self.feature_names) if name in temporal_features]

        if temporal_indices:
            temporal_data = features_normalized[:, temporal_indices]
            temporal_variance = np.mean(np.var(temporal_data, axis=0))
            self.results['diversity_metrics']['temporal_diversity'] = float(temporal_variance)
            print(f"   Temporal diversity: {temporal_variance:.3f}")

        # 6. Composite Diversity Index (CDI)
        print("\n[6] Composite Diversity Index...")
        # Weighted combination of multiple diversity measures
        cdi = (
                0.3 * (overall_variance / 2.0) +  # Normalized overall variance
                0.2 * (n_components_95 / 10.0) +  # Normalized dimensionality
                0.3 * (mean_distance / 5.0) +  # Normalized mean distance
                0.1 * (spectral_variance if spectral_indices else 0) +
                0.1 * (temporal_variance if temporal_indices else 0)
        )

        cdi = min(1.0, max(0.0, cdi))  # Clip to [0, 1]

        self.results['diversity_metrics']['composite_diversity_index'] = float(cdi)
        print(f"   Composite Diversity Index: {cdi:.3f}")
        print(f"   (0 = uniform sounds, 1 = highly diverse)")

    def identify_sound_clusters(self):
        """Identify clusters of similar percussion sounds"""
        print("\n[7] Sound Clustering Analysis...")

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.features_matrix)

        # Try different numbers of clusters
        best_k = 2
        best_score = -1

        for k in range(2, min(10, len(self.onset_segments) // 2)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_normalized)
            score = silhouette_score(features_normalized, labels)

            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)

        # Count sounds in each cluster
        unique, counts = np.unique(labels, return_counts=True)

        self.results['diversity_metrics']['num_sound_types'] = int(best_k)
        self.results['diversity_metrics']['sound_distribution'] = counts.tolist()
        self.results['diversity_metrics']['silhouette_score'] = float(best_score)

        print(f"   Identified {best_k} distinct sound types")
        print(f"   Sound distribution: {counts}")
        print(f"   Cluster quality (silhouette): {best_score:.3f}")

    def generate_summary(self):
        """Generate human-readable summary"""
        print("\n" + "=" * 60)
        print("PERCUSSION DIVERSITY SUMMARY")
        print("=" * 60)

        dm = self.results['diversity_metrics']

        print(f"\nSound Collection:")
        print(f"  • Total drum hits analyzed: {self.results['num_segments']}")
        print(f"  • Distinct sound types: {dm.get('num_sound_types', 'N/A')}")

        print(f"\nDiversity Metrics:")
        print(f"  • Composite Diversity Index: {dm['composite_diversity_index']:.3f} / 1.0")
        print(f"  • Overall variance: {dm['overall_variance']:.3f}")
        print(f"  • Mean timbre distance: {dm['mean_pairwise_distance']:.3f}")
        print(f"  • Spectral diversity: {dm.get('spectral_diversity', 0):.3f}")
        print(f"  • Temporal diversity: {dm.get('temporal_diversity', 0):.3f}")

        print(f"\nTimbre Space:")
        print(f"  • Effective dimensions: {dm['pca_components_95']}")
        print(f"  • Cluster quality: {dm.get('silhouette_score', 0):.3f}")

        print(f"\nInterpretation:")
        cdi = dm['composite_diversity_index']

        if cdi > 0.7:
            print("  • HIGHLY DIVERSE: Rich variety of percussion timbres")
            print("    (Complex drum kit, many instruments, varied playing techniques)")
        elif cdi > 0.5:
            print("  • MODERATELY DIVERSE: Good timbral variety")
            print("    (Standard drum kit with some variation)")
        elif cdi > 0.3:
            print("  • LIMITED DIVERSITY: Some variation present")
            print("    (Basic drum kit or electronic drums with limited sounds)")
        else:
            print("  • LOW DIVERSITY: Mostly uniform sounds")
            print("    (Single drum or very consistent electronic percussion)")

        # Spectral characteristics
        print(f"\nSpectral Characteristics:")
        feats = self.results['features']

        centroid_mean = feats['spectral_centroid']['mean']
        if centroid_mean > 3000:
            print(f"  • Bright sounds (centroid: {centroid_mean:.0f} Hz)")
        elif centroid_mean > 1500:
            print(f"  • Moderate brightness (centroid: {centroid_mean:.0f} Hz)")
        else:
            print(f"  • Dark sounds (centroid: {centroid_mean:.0f} Hz)")

        flatness_mean = feats['spectral_flatness']['mean']
        if flatness_mean > 0.5:
            print("  • Noisy/metallic character (high flatness)")
        else:
            print("  • Tonal character (low flatness)")

        # Temporal characteristics
        print(f"\nTemporal Characteristics:")
        attack_mean = feats['attack_time']['mean']
        decay_mean = feats['decay_time']['mean']

        print(f"  • Average attack time: {attack_mean * 1000:.1f} ms")
        print(f"  • Average decay time: {decay_mean * 1000:.1f} ms")

        if attack_mean < 0.01:
            print("  • Very sharp attacks (typical of drums)")

        if decay_mean > 0.2:
            print("  • Long sustain (cymbals, resonant drums)")
        elif decay_mean < 0.05:
            print("  • Short, punchy sounds (tight gated drums)")

        print("\n" + "=" * 60)

    def save_results(self, output_file='percussion_diversity_analysis.txt'):
        """Save detailed results to file"""
        with open(output_file, 'w') as f:
            f.write("PERCUSSION DIVERSITY ANALYSIS - DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")

            # Write diversity metrics
            f.write("DIVERSITY METRICS:\n")
            f.write("-" * 60 + "\n")
            for key, value in self.results['diversity_metrics'].items():
                if isinstance(value, (list, dict)):
                    f.write(f"{key}:\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"  {value}\n")
                else:
                    f.write(f"{key}: {value}\n")

            # Write feature statistics
            f.write("\n\nFEATURE STATISTICS:\n")
            f.write("-" * 60 + "\n")
            for feature, stats in self.results['features'].items():
                f.write(f"\n{feature}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value:.6f}\n")

        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "trimmed_output33.wav"  # Change this to your WAV file name

    print("=" * 60)
    print("PERCUSSION DIVERSITY OF SOUND ANALYZER")
    print("=" * 60)
    print("\nThis tool analyzes the diversity of percussion timbres using:")
    print("• Spectral features (brightness, spread, harmonicity)")
    print("• Temporal features (attack, decay, envelope)")
    print("• Statistical diversity metrics")
    print("• Timbre space dimensionality")
    print()

    # Initialize analyzer
    analyzer = PercussionDiversityAnalyzer(AUDIO_FILE)

    # Step 1: Separate drums using Demucs
    analyzer.separate_drums()

    # Step 2: Extract individual drum hit segments
    analyzer.extract_onset_segments()

    # Step 3: Extract spectral and temporal features
    analyzer.extract_all_features()

    # Step 4: Calculate diversity metrics
    analyzer.calculate_diversity_metrics()

    # Step 5: Identify sound clusters
    analyzer.identify_sound_clusters()

    # Step 6: Generate summary
    analyzer.generate_summary()

    # Step 7: Save results
    analyzer.save_results()

    print("\n✓ Analysis complete!")
    print("\nYou now have quantified:")
    print("  • Spectral diversity (brightness, harmonicity, noise)")
    print("  • Temporal diversity (attack/decay patterns)")
    print("  • Overall timbre diversity (composite index)")
    print("  • Number of distinct percussion sounds")


    #composite_diversity_index (Let's save this)