# tests/test_classification.py
import unittest
import numpy as np
from src.classify import EEGClassifier
import os

class TestClassifier(unittest.TestCase):
    def setUp(self):
        # Sample configuration for the classifier
        self.config = {
            'classification': {
                'model': 'svm',
                'kernel': 'rbf'
            }
        }
        self.model_path = 'models/test_classifier_model.pkl'
        self.classifier = EEGClassifier(self.config, model_path=self.model_path)
        
        # Sample data for testing
        self.X_train = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ])  # 3 samples, 4 features
        self.y_train = np.array([0, 1, 0])  # Binary labels (left/right)

    def test_train_and_save(self):
        """Test training and saving the classifier model."""
        self.classifier.train(self.X_train, self.y_train)
        self.assertTrue(os.path.exists(self.model_path), "Model file was not saved.")

    def test_predict(self):
        """Test classifier prediction."""
        self.classifier.train(self.X_train, self.y_train)
        test_feature = np.array([1.0, 2.0, 3.0, 4.0])
        prediction = self.classifier.predict(test_feature)
        self.assertIn(prediction, [0, 1], "Prediction should be 0 or 1.")

    def test_load_model(self):
        """Test loading a saved model."""
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.load_model()
        test_feature = np.array([1.0, 2.0, 3.0, 4.0])
        prediction = self.classifier.predict(test_feature)
        self.assertIn(prediction, [0, 1], "Loaded model should predict 0 or 1.")

    def tearDown(self):
        """Clean up by removing the test model file."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()