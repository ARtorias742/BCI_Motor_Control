# src/feature_extraction.py
import mne
import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        self.sampling_rate = config['eeg']['sampling_rate']
        self.freq_bands = config['feature_extraction']['freq_bands']
        self.channels = config['eeg']['channels']

    def extract_features(self, data):
        """Extract PSD features for each channel and frequency band."""
        info = mne.create_info(ch_names=self.channels, sfreq=self.sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        features = []
        for band, (low, high) in self.freq_bands.items():
            # Compute PSD for the frequency band
            psds, _ = mne.time_frequency.psd_welch(raw, fmin=low, fmax=high, n_fft=256)
            # Average PSD across time for each channel
            psd_mean = psds.mean(axis=1)
            features.append(psd_mean)

        return np.concatenate(features)  # Shape: (n_channels * n_bands,)