# src/preprocess.py
import mne
import numpy as np

class EEGPreprocessor:
    def __init__(self, config):
        self.sampling_rate = config['eeg']['sampling_rate']
        self.low_freq = config['preprocessing']['low_freq']
        self.high_freq = config['preprocessing']['high_freq']
        self.notch_freq = config['preprocessing']['notch_freq']
        self.channels = config['eeg']['channels']

    def preprocess(self, raw_data):
        """Preprocess EEG data: bandpass and notch filtering."""
        # Create MNE Raw object
        info = mne.create_info(ch_names=self.channels, sfreq=self.sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(raw_data, info)

        # Apply notch filter (remove power line noise)
        raw.notch_filter(self.notch_freq)

        # Apply bandpass filter (alpha and beta bands)
        raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)

        return raw.get_data()