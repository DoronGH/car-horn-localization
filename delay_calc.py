import numpy as np


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
    correlation = np.correlate(signal1, signal2, "full")

    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(correlation)

    # Calculate the delay in samples
    delay_samples = max_corr_index - (len(signal1) - 1)

    # Convert the delay from samples to seconds
    delay_seconds = delay_samples / fs

    return delay_seconds


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

    # Calculate the delay
    calculated_delay = calculate_delay(signal1, signal2, fs)
    print(f"Calculated delay: {calculated_delay} seconds")
