import noisereduce as nr
import numpy as np
from matplotlib import pyplot as plt

F_MIN = 500
F_MAX = 5000
MIN_RATIO = 0.1
TOP_VAL_RATIO = 3000


# DETECTION FUNCTIONS #

def compute_energy(signal):
    """
    Computes the energy of a signal
    :param signal: the signal
    :return: the energy of the signal
    """
    energy = np.sum(np.abs(signal) ** 2)
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


def horn_classification(signal, fs):
    """
    Detects the presence of a horn-like sound in an audio signal.
    :param signal: The audio signal.
    :param fs: The sample rate of the audio signal.
    :return: True if a horn-like sound is detected, False otherwise.
    """
    filtered_signal = nr.reduce_noise(y=signal, sr=fs)
    fourier_signal = np.fft.fft(filtered_signal)
    fourier_signal_energy = compute_energy(fourier_signal)
    strong_freqs_energy = compute_strong_freqs_energy(fourier_signal, fs)
    if strong_freqs_energy / fourier_signal_energy >= MIN_RATIO:
        return True
    return False


def detect_horn(signal, fs):
    sub_arrays = split_audio_array(signal, int(fs / 2))
    detections = []
    for i, sub_array in enumerate(sub_arrays):
        classification = horn_classification(sub_array, fs)
        detections.append(classification)
        # plot_fft(sub_array, fs, i, classification)
    return detections
