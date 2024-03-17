# Car Horn Detection and Localization System

Car Horn Detection and Localization is a project aimed at identifying and mitigating noise pollution in urban environments, specifically focusing on car horn sounds. Our system uses a combination of audio processing and visual data analysis to detect, localize, and identify vehicles responsible for excessive honking.

## Components

This repository consists of several Python files crucial to the operation of the Urban Silence project:

- **horn_detector.py**: Contains algorithms for detecting car horn sounds using audio data. It utilizes band-pass filtering and FFT (Fast Fourier Transform) to isolate and identify horn sounds.

- **horn_localization.py**: Implements the logic for localizing the source of detected car horn sounds using an array of microphones. This involves calculating time delays and using cross-correlation techniques.

- **main.py**: The entry point of the project, orchestrating the detection and localization processes. It includes functions for reading audio files, processing signals, and integrating the horn detection and localization functionalities.

- **read_mic.py**: Provides utilities for recording audio from microphones, essential for real-time data capture and system testing.

- **syncronizer.py**: Focuses on synchronizing audio signals from multiple sources, critical for accurate sound localization.

## Setup

To run this project, you will need:

1. Python 3.x installed on your system.
2. The following Python packages: `numpy`, `matplotlib`, `scipy`, `noisereduce`, `pydub`, and `pyaudio`. Install them using pip: 'pip install numpy matplotlib scipy noisereduce pydub pyaudio'


## Usage

1. **Recording Audio:**
- Use `read_mic.py` to record audio from your selected microphones. Adjust the microphone indices in the script to match your setup.

2. **Detecting and Localizing Horn Sounds:**
- Run `main.py` to execute the horn detection and localization process on your recorded audio. The script reads audio files, processes the signals, and applies the detection and localization algorithms.

3. **Adjusting Parameters:**
- You can tweak detection and localization parameters in `horn_detector.py` and `horn_localization.py` to match the characteristics of your environment or to experiment with different settings.
