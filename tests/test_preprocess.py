# tests/test_preprocess.py
import unittest
import numpy as np
from src.preprocess import EEGPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'eeg': {'sampling_rate': 250, 'channels': ['C3', 'C4']},
            'preprocessing': {'low_freq': 8, 'high_freq': 30, 'notch_freq': 50}
        }
        self.preprocessor = EEGPreprocessor(self.config)

    def test_preprocess(self):
        data = np.random.randn(2, 250)  # 2 channels, 1 second of data
        processed = self.preprocessor.preprocess(data)
        self.assertEqual(processed.shape, data.shape)

if __name__ == '__main__':
    unittest.main()