from matplotlib import pyplot as plt
import noisereduce as nr
import numpy as np


def plot_fft(signal, sample_rate, i):
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
    plt.title(f'FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([-20000, 20000])
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]


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