# src/classify.py
from sklearn.svm import SVC
import joblib
import numpy as np

class EEGClassifier:
    def __init__(self, config, model_path='models/classifier_model.pkl'):
        self.model_type = config['classification']['model']
        self.kernel = config['classification']['kernel']
        self.model_path = model_path
        self.model = SVC(kernel=self.kernel, probability=True)

    def train(self, X, y):
        """Train the classifier on feature data."""
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def predict(self, X):
        """Predict motor imagery class (e.g., 0: left, 1: right)."""
        return self.model.predict(X.reshape(1, -1))[0]

    def load_model(self):
        """Load a pre-trained model."""
        self.model = joblib.load(self.model_path)