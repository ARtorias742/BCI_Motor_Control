# src/train_classifier.py
import numpy as np
from sklearn.svm import SVC
import joblib
import yaml
from feature_extraction import FeatureExtractor
from preprocess import EEGPreprocessor
from acquire_data import EEGDataAcquisition

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_simulated_training_data(config, n_samples=100):
    """Generate simulated EEG data and labels for training."""
    acquisition = EEGDataAcquisition(config, use_simulated=True)
    preprocessor = EEGPreprocessor(config)
    extractor = FeatureExtractor(config)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Simulate EEG data (2 channels)
        raw_data = acquisition.acquire()
        
        # Preprocess
        preprocessed_data = preprocessor.preprocess(raw_data)
        
        # Extract features
        features = extractor.extract_features(preprocessed_data)
        X.append(features)
        
        # Assign random labels (0: left, 1: right) for simulation
        label = np.random.choice([0, 1])
        y.append(label)
    
    return np.array(X), np.array(y)

def train_model(config):
    """Train the SVM classifier and save it."""
    # Generate or load training data
    X, y = generate_simulated_training_data(config)
    
    # Initialize classifier
    model_path = 'models/classifier_model.pkl'
    classifier = SVC(kernel=config['classification']['kernel'], probability=True)
    
    # Train
    classifier.fit(X, y)
    
    # Save model
    joblib.dump(classifier, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Ensure models/ directory exists
    import os
    os.makedirs('models', exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Train and save the model
    train_model(config)