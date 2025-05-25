# BCI Motor Control Prototype

A brain-computer interface for controlling a virtual cursor using EEG motor imagery signals.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure parameters in `config/config.yaml`.
3. (Optional) Connect OpenBCI hardware and update port in `src/acquire_data.py`.
4. Run: `python src/main.py`

## Usage
- The system acquires EEG data (simulated or real).
- Signals are preprocessed, features extracted, and classified as left or right motor imagery.
- A cursor moves left or right on a Pygame window based on predictions.

## Notes
- Train the classifier with labeled EEG data for real use.
- Extend to physical devices (e.g., robotic arm) using Raspberry Pi GPIO.