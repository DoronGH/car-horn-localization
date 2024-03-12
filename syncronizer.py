import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate, fftconvolve, lfilter
from scipy.fft import fft, ifft

from gccestimating import GCC

CORR_LEN = 7


def compute_delay(signal1, signal2, i):
    """
    Compute the delay between two signals using cross-correlation.

    Parameters:
    - signal1: NumPy array, the first signal
    - signal2: NumPy array, the second signal
    - fs: float, the sampling frequency of the signals

    Returns:
    - delay: float, the delay between the two signals in seconds
    """
    signal1 = signal1.astype(np.float64)
    signal2 = signal2.astype(np.float64)

    # Normalize signals
    signal1 /= np.max(np.abs(signal1))
    signal2 /= np.max(np.abs(signal2))

    # Compute cross-correlation
    cross_corr = correlate(signal1, signal2, mode='full')

    # Find the index of the maximum correlation
    delay_index = np.argmax(cross_corr)  # used to synchronize signals

    # Calculate the delay in seconds
    samples_delay = (delay_index - len(signal1) + 1)

    # print("signal1.shape: ", signal1.shape)
    # print("signal2.shape: ", signal2.shape)
    # print("correlation.shape: ", cross_corr.shape)
    # plt.figure()
    # plt.plot(cross_corr)
    # plt.grid(True)
    # # plt.xlim([delay_index-1000, delay_index+1000])
    # plt.title(f'index = {i}')
    # plt.show()

    return samples_delay


def sync(signal1, signal2, fs):
    time_len = len(signal1) // fs
    range_size = (time_len - CORR_LEN) * 2 + 1
    for n in range(range_size):
        print(compute_delay(signal1[int(0.5 * n) * fs:int(CORR_LEN + 0.5 * n) * fs], signal2[int(0.5 * n) * fs:int(CORR_LEN + 0.5 * n) * fs], n))
