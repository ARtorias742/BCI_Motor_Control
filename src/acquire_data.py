# src/acquire_data.py
import numpy as np
from openbci import OpenBCIBoard  # Optional, for real OpenBCI hardware
import time

class EEGDataAcquisition:
    def __init__(self, config, use_simulated=True):
        self.sampling_rate = config['eeg']['sampling_rate']
        self.channels = config['eeg']['channels']
        self.use_simulated = use_simulated
        self.board = None
        if not use_simulated:
            self.board = OpenBCIBoard(port='COM3')  # Adjust port for your setup

    def simulate_eeg(self):
        """Simulate EEG data for testing (2 channels, alpha/beta bands)."""
        t = np.linspace(0, 1, self.sampling_rate)
        alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha wave
        beta = np.sin(2 * np.pi * 20 * t)   # 20 Hz beta wave
        noise = np.random.normal(0, 0.1, t.shape)
        data = np.vstack([alpha + noise, beta + noise])  # 2 channels
        return data

    def acquire(self):
        """Acquire EEG data (simulated or real)."""
        if self.use_simulated:
            return self.simulate_eeg()
        else:
            # Stream real data from OpenBCI
            data = []
            def handle_sample(sample):
                data.append(sample.channels_data)
            self.board.start_streaming(handle_sample)
            time.sleep(1)  # Collect 1 second of data
            self.board.stop_streaming()
            return np.array(data).T  # Shape: (channels, samples)