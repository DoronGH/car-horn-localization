# Urban Silence: Car Horn Detection and Localization System

Urban Silence is a comprehensive system designed to combat noise pollution in urban environments. The project focuses on the detection, localization, and identification of vehicles responsible for excessive honking. The system leverages audio processing techniques and visual data analysis to achieve its goals.

## Project Structure

The project is structured into several Python scripts, each serving a specific purpose:

- **main.py**: This is the entry point of the project. It orchestrates the detection and localization processes, integrates the functionalities of the other scripts, and includes functions for reading audio files and processing signals.

- **horn_detector.py**: This script is responsible for the detection of car horn sounds. It employs techniques such as band-pass filtering and Fast Fourier Transform (FFT) to isolate and identify horn sounds from audio data.

- **horn_localization.py**: This script localizes the source of the detected car horn sounds. It calculates time delays and uses cross-correlation techniques to pinpoint the location of the sound source.

- **read_mic.py**: This script provides utilities for recording audio from microphones, which is essential for real-time data capture and system testing.

- **syncronizer.py**: This script is responsible for synchronizing audio signals from multiple sources, which is crucial for accurate sound localization.

- **utils.py**: This script contains utility functions for plotting FFT and energy ratio.

- **visual_detection.py**: This script implements vehicle detection using visual data analysis. It identifies vehicles in video frames and extracts license plate information.

## Setup

To run this project, you will need:

1. Python 3.x installed on your system.
2. The following Python packages: `numpy`, `matplotlib`, `scipy`, `noisereduce`, `pydub`, `pyaudio`, and `opencv-python`. Install them using pip:
   `pip install numpy matplotlib scipy noisereduce pydub pyaudio opencv-python`

## Usage

1. **Setting Up Data Paths:**
- Before running the project, you need to specify the paths to your audio and video data. These paths are defined in the `main.py` file as `AUDIO1_PATH`, `AUDIO2_PATH`, and `VIDEO_PATH`. Replace the existing paths with the paths to your own data files.

2. **Running the Project:**
- Execute `main.py` to start the project. This script orchestrates the detection and localization of car horn sounds and integrates the functionalities of the other scripts.

3. **Adjusting Parameters:**
- You can modify detection and localization parameters in `horn_detector.py` and `horn_localization.py` to suit the characteristics of your environment or to experiment with different settings.

4. **Visual Detection:**
- The `main.py` script also integrates with `visual_detection.py` to detect vehicles in video frames and extract license plate information. you can edit line 34 in this file to enable or disable the license plate detection.