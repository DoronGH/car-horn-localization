
import numpy as np
import noisereduce as nr
import torch

F_MAX = 2000
MAX_FREQ_NUM = 20
ENERGY_THRESHOLD = 0
MIN_RATIO = 0
TOP_VAL_RATIO = 5000


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
    max_freq_idx = np.int64((F_MAX * N) / sample_rate)
    low_freqs = np.abs(fourier_signal[0:max_freq_idx + 1])

    # Find the strongest frequencies
    strongest_freqs = np.zeros(int(N / TOP_VAL_RATIO))
    for i in range(int(N / TOP_VAL_RATIO)):
        max_index = np.argmax(low_freqs)
        strongest_freqs[i] = (low_freqs[max_index])
        low_freqs[max_index] = -np.inf

    return compute_energy(strongest_freqs)


def horn_detect(signal, sample_rate):
    """
    Detects the presence of a horn-like sound in an audio signal.
    :param signal: The audio signal.
    :param sample_rate: The sample rate of the audio signal.
    :return: True if a horn-like sound is detected, False otherwise.
    """
    signal_energy = compute_energy(signal)
    if signal_energy >= ENERGY_THRESHOLD:
        filtered_signal = nr.reduce_noise(y=signal, sr=sample_rate)
        fourier_signal = np.fft.fft(filtered_signal)
        fourier_signal_energy = compute_energy(fourier_signal)
        strong_freqs_energy = compute_strong_freqs_energy(fourier_signal, sample_rate)
        if strong_freqs_energy / fourier_signal_energy >= MIN_RATIO:
            print(strong_freqs_energy / fourier_signal_energy)
            return True
    return False

