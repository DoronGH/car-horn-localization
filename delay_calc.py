import numpy as np

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

    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(correlation)

    # Calculate the delay in samples
    delay_samples = max_corr_index - (len(signal1)//2)

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

    # # Calculate the delay
    # calculated_delay = calculate_delay(signal1, signal2, fs)
    # print(f"Calculated delay: {calculated_delay} seconds")

    # Generate example signals (for demonstration purposes)
    fs = 44100  # Sampling rate in Hz
    frequency = 5  # Frequency of the sine wave in Hz
    signal1_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_b_mic_1.wav"
    signal2_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_b_mic_4.wav"
    fs, signal1 = read_wav_file(signal1_path)
    _, signal2 = read_wav_file(signal2_path)
    signal2_new = signal2[22000:66000]

    # modified_signal = np.zeros_like(signal2)
    # modified_signal[start_idx:end_idx] = signal[start_idx:end_idx]

    # Calculate the delay
    calculated_delay = calculate_delay(signal1, signal2_new, fs)
    print(f"Calculated delay: {calculated_delay} seconds")
    print(fs)

    # import matplotlib.pyplot as plt
    #
    # # n_repeats = 100
    # n = 1000
    # n_repeats = 1
    # # Get correlations
    # t = np.linspace(0, n_repeats, n)
    # sin_delay = lambda delay: np.sin(2.0 * np.pi * (t - delay))
    # signal1 = sin_delay(delay=0)
    # signal2 = sin_delay(delay=x)
    # corr11 = signal.correlate(signal1, signal1, mode='full')
    # corr12 = signal.correlate(signal1, signal2, mode='full')
    # a1 = np.argmax(corr11)
    # a2 = np.argmax(corr12)
    # # Print output
    # print(a1, a2, x, n_repeats * (a1 - a2) / n)
    # # Make plots
    # plt.figure()
    # plt.plot(signal1, "r")
    # plt.plot(signal2, "b")
    # plt.title("Signals, delay = {:.3f}".format(x))
    # plt.legend(["Original signal", "Delayed signal"], loc="upper right")
    # plt.grid(True)
    # plt.savefig("Signals")
    # plt.figure()
    # plt.plot(corr11, "r")
    # plt.plot(corr12, "b")
    # plt.title("Correlations, delay = {:.3f}".format(x))
    # plt.legend(["Auto-correlation", "Cross-correlation"], loc="upper right")
    # plt.grid(True)
    # plt.savefig("Correlations")