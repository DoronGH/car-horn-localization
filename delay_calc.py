import numpy as np
import scipy

from scipy.io import wavfile


def read_wav_file(filename):
    """
    Read a WAV file and return its sample rate and data.

    Parameters:
    - filename: The path to the WAV file.

    Returns:
    - fs: The sample rate of the WAV file (in samples per second).
    - data: The data read from the WAV file as a NumPy array.
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(filename)

    return sample_rate, data

def calculate_delay(signal1, signal2, fs):
    """
    Calculate the delay between two signals using cross-correlation.

    Parameters:
    - signal1: The first signal (numpy array).
    - signal2: The second signal (numpy array), should be of the same length as signal1.
    - fs: The sampling rate of the signals.

    Returns:
    - delay: The calculated delay between the two signals in seconds.
    """
    # Ensure the signals are numpy arrays
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    # Compute the cross-correlation between the two signals
    correlation = np.correlate(signal1, signal2, "same")
    correlation = scipy.signal.correlate(signal1, signal2, "same")

    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(correlation)

    # Calculate the delay in samples
    delay_samples = max_corr_index - (len(signal1)//2)

    # Convert the delay from samples to seconds
    delay_seconds = delay_samples / fs

    return delay_seconds

import matplotlib.pyplot as plt

def calculate_signal_delay(signal1, signal2, fs):
    """
    Calculate the delay between two signals using cross-correlation.

    Parameters:
    - signal1: The first signal (numpy array).
    - signal2: The second signal (numpy array), should be of the same length as signal1 for accurate results.
    - fs: The sampling rate of the signals in Hz.

    Returns:
    - delay: The calculated delay between the two signals in seconds. A positive delay means signal2 lags signal1.
    """

    # Compute the cross-correlation between the two signals
    # correlation = np.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
    correlation = scipy.signal.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='valid')
    # correlation = scipy.signal.correlate(signal1, signal2, mode='valid')
    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(correlation)

    # Calculate the lag in samples. The peak of the cross-correlation gives the index of maximum similarity.
    # Adjusting by the length of signal1 to find the actual lag.
    # lag_samples = max_corr_index - len(signal1) + 1

    lag_samples = max_corr_index - len(correlation)//2

    # Convert the lag from samples to seconds
    delay_seconds = lag_samples / fs
    print("signal1.shape: ", signal1.shape)
    print("signal2.shape: ", signal2.shape)
    print("correlation.shape: ", correlation.shape)
    plt.figure()
    plt.plot(correlation)
    plt.grid(True)
    plt.show()

    return delay_seconds


from pydub import AudioSegment


def read_m4a(file_path):
    audio = AudioSegment.from_file(file_path, format="m4a")
    return audio


# Example usage:
if __name__ == "__main__":

    # Generate example signals (for demonstration purposes)
    fs = 44100  # Sampling rate in Hz
    t = np.linspace(0, 1, fs, endpoint=False)
    frequency = 5  # Frequency of the sine wave in Hz
    signal1 = np.sin(2 * np.pi * frequency * t)
    delay_time = 0.1  # Delay in seconds
    delay_samples = int(delay_time * fs)
    signal2 = np.pad(signal1[:-delay_samples], (delay_samples, 0), 'constant')

    # # Calculate the delay
    # calculated_delay = calculate_delay(signal1, signal2, fs)
    # print(f"Calculated delay: {calculated_delay} seconds")

    # Generate example signals (for demonstration purposes)
    fs = 44100  # Sampling rate in Hz
    frequency = 5  # Frequency of the sine wave in Hz
    signal1_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_h_mic_1.wav"
    signal2_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_h_mic_2.wav"
    signal4_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_h_mic_4.wav"

    fs, signal1 = read_wav_file(signal1_path)
    _, signal2 = read_wav_file(signal2_path)
    _, signal4 = read_wav_file(signal4_path)
    add_delay = 0
    start_time_arr = range(0,15,3)
    delay_1_2 = []
    delay_1_4 = []
    for start_time in start_time_arr:
        signal1_new = signal1[start_time * fs: (start_time + 5) * fs]
        signal2_new = signal2[int((start_time + 1.5 + add_delay) * fs): int((start_time + 3.5 + add_delay) * fs)]
        signal4_new = signal4[int((start_time + 1.5 + add_delay) * fs): int((start_time + 3.5 + add_delay) * fs)]

        # Calculate the delay
        calculated_delay = calculate_signal_delay(signal1_new, signal2_new, fs)
        delay_1_2.append(calculated_delay)
        # print(f"Calculated delay: {calculated_delay} seconds")
        calculated_delay = calculate_signal_delay(signal1_new, signal4_new, fs)
        delay_1_4.append(calculated_delay)
        # print(f"Calculated delay: {calculated_delay} seconds")

    print(delay_1_2)
    print(delay_1_4)

    """
    Calculated delay: -0.0006575963718820862 seconds
    44100
    Calculated delay: -0.0014285714285714286 seconds
    
    
    3-6
    Calculated delay: 0.0014965986394557824 seconds
    Calculated delay: 0.0008616780045351474 seconds
    """