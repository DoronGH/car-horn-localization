import numpy as np
import noisereduce as nr
import torch
from matplotlib import pyplot as plt

F_MIN = 500
F_MAX = 5000
MAX_FREQ_NUM = 20
ENERGY_THRESHOLD = 0
MIN_RATIO = 0.05
TOP_VAL_RATIO = 2500


def plot_fft(signal, sample_rate, i, ratio):
    # Apply FFT to the signal
    filtered_signal = nr.reduce_noise(y=signal, sr=sample_rate)
    fft_result = np.fft.fft(filtered_signal)

    # Calculate the frequency values corresponding to the FFT result
    freq_values = np.fft.fftfreq(len(signal), 1 / sample_rate)

    # Shift the zero frequency component to the center
    fft_result_shifted = np.fft.fftshift(fft_result)
    freq_values_shifted = np.fft.fftshift(freq_values)

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq_values_shifted, np.abs(fft_result_shifted))
    plt.title(f'FFT, {i}, Ratio = {ratio}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([0, 2000])
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]


def compute_energy(signal):
    """
    Computes the energy of a signal
    :param signal: the signal
    :return: the energy of the signal
    """
    energy = np.sum(np.abs(signal)**2)
    return energy


def compute_strong_freqs_energy(fourier_signal, sample_rate):
    """
    Computes the energy of the strongest frequencies in a Fourier-transformed signal.
    :param fourier_signal: The Fourier-transformed signal.
    :param sample_rate: The sample rate of the original signal.
    :return: The energy of the strongest frequencies.
    """

    # Take only the relevant frequencies
    N = len(fourier_signal)
    min_freq_idx = np.int64((F_MIN * N) / sample_rate)
    max_freq_idx = np.int64((F_MAX * N) / sample_rate)
    band_pass = np.abs(fourier_signal[min_freq_idx:max_freq_idx + 1])

    # Find the strongest frequencies
    strongest_freqs = np.zeros(int(N / TOP_VAL_RATIO))
    for i in range(int(N / TOP_VAL_RATIO)):
        max_index = np.argmax(band_pass)
        strongest_freqs[i] = (band_pass[max_index])
        band_pass[max_index] = -np.inf

    return compute_energy(strongest_freqs)


def split_audio_array(signal, n):
    sub_arrays = []
    for i in range(0, len(signal), n):
        subarray = signal[i:i + n]
        sub_arrays.append(subarray)
    return sub_arrays


def detect_horn(signal, sample_rate):
    sub_arrays = split_audio_array(signal, sample_rate)
    detections = []
    for i, sub_array in enumerate(sub_arrays):
        ratio = horn_classification(sub_array, sample_rate)
        detections.append(ratio)
#        plot_fft(sub_array, sample_rate, i, ratio)
    return detections

def plot_energy_and_ratio(x, energy, ratio):
    """
    Plot three arrays: x, energy, and ratio.

    Parameters:
    - x: Array representing the x-axis values.
    - energy: Array representing the energy (y-axis) values.
    - ratio: Array representing the ratio (y-axis) values.

    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=(10, 6))

    # Plot energy
    plt.subplot(2, 1, 1)
    plt.plot(x, energy, label='Energy')
    plt.title('Energy Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Energy')

    # Plot ratio
    plt.subplot(2, 1, 2)
    plt.plot(x, ratio, "*", label='Ratio')
    plt.title('Ratio Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Ratio')

    plt.tight_layout()
    plt.show()


def horn_classification(signal, sample_rate):
    """
    Detects the presence of a horn-like sound in an audio signal.
    :param signal: The audio signal.
    :param sample_rate: The sample rate of the audio signal.
    :return: True if a horn-like sound is detected, False otherwise.
    """
    filtered_signal = nr.reduce_noise(y=signal, sr=sample_rate)
    fourier_signal = np.fft.fft(filtered_signal)
    fourier_signal_energy = compute_energy(fourier_signal)
    strong_freqs_energy = compute_strong_freqs_energy(fourier_signal, sample_rate)
    if strong_freqs_energy / fourier_signal_energy >= MIN_RATIO:
        return True
    return False
