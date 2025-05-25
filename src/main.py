# src/main.py
import yaml
from acquire_data import EEGDataAcquisition
from preprocess import EEGPreprocessor
from feature_extraction import FeatureExtractor
from classify import EEGClassifier
from visualize import CursorVisualizer

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Initialize components
    acquisition = EEGDataAcquisition(config, use_simulated=True)  # Set to False for real EEG
    preprocessor = EEGPreprocessor(config)
    extractor = FeatureExtractor(config)
    classifier = EEGClassifier(config)
    visualizer = CursorVisualizer(config)

    # Load or train classifier (for demo, assume pre-trained model or simulated labels)
    # For real use, train with labeled EEG data: classifier.train(X_train, y_train)
    classifier.load_model()  # Assumes a pre-trained model exists

    # Main loop
    running = True
    while running:
        # Acquire EEG data
        raw_data = acquisition.acquire()

        # Preprocess
        preprocessed_data = preprocessor.preprocess(raw_data)

        # Extract features
        features = extractor.extract_features(preprocessed_data)

        # Classify (0: left, 1: right)
        command = classifier.predict(features)

        # Update visualization
        visualizer.update(command)

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    visualizer.quit()

if __name__ == '__main__':
    main()