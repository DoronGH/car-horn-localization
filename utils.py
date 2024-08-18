from matplotlib import pyplot as plt
import noisereduce as nr
import numpy as np


def plot_fft(signal, sample_rate, i):
    """
    Apply FFT to the signal and plot the magnitude spectrum.
    :param signal: NumPy array, the audio signal
    :param sample_rate: float, the sample rate of the audio signal
    :param i: list, contains the current time and a boolean indicating if a detection was found
    :return: list, contains the FFT result and the corresponding frequency values
    """
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
    plt.title(f'FFT\nTime: {i[0]}, Detection Found: {i[1]}', fontsize=16)
    plt.xlim([-20000, 20000])
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]


def plot_energy_and_ratio(x, energy, ratio):
    """
    Plot three arrays: x, energy, and ratio.
    :param x: Array representing the x-axis values.
    :param energy: Array representing the energy (y-axis) values.
    :param ratio: Array representing the ratio (y-axis) values.
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
