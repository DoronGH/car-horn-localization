
import numpy as np

F_MAX = 1500
MAX_FREQ_NUM = 10
ENERGY_THRESHOLD = 0
MIN_RATIO = 0.1


def compute_energy(signal):
    energy = np.sum(np.abs(signal)**2)
    return energy


def compute_strong_freqs_energy(signal, sample_rate):
    fft_result = np.fft.fft(signal)
    # print(compute_energy(signal))
    # print(compute_energy(fft_result) / len(fft_result))
    max_freq_idx = np.int64((F_MAX * len(signal)) / sample_rate)
    low_freqs = np.abs(fft_result[0:max_freq_idx + 1])
    top_values = np.zeros(MAX_FREQ_NUM)

    for i in range(MAX_FREQ_NUM):
        # Find the index of the maximum element
        max_index = np.argmax(low_freqs)

        # Append the maximum value to the result list
        top_values[i] = (low_freqs[max_index])

        # Set the maximum element to a very small value
        low_freqs[max_index] = -np.inf

    return np.sum(np.square(top_values))


def horn_detect(signal, sample_rate):
    signal_energy = compute_energy(signal)
    if signal_energy >= ENERGY_THRESHOLD:
        strong_freqs_energy = compute_strong_freqs_energy(signal, sample_rate)
        if strong_freqs_energy / signal_energy >= MIN_RATIO:
            return True
    return False

