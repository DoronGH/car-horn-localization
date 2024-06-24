import numpy as np
from scipy.signal import butter, filtfilt, correlate


SPEED_OF_SOUND = 343.2
DIST = 3.007
HIGH_PASS_CUTOFF = 3000
DELAY_PER_SEC = 0.65


def high_pass_filter(signal, sample_rate, cutoff):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist

    # Design high-pass Butterworth filter
    b, a = butter(N=4, Wn=normal_cutoff, btype='high', analog=False)

    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def compute_delay(signal1, signal2, fs, sync_time, curr_time):
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
    mid_index = len(cross_corr) // 2 + 1
    delta = int((DIST * fs) / SPEED_OF_SOUND)
    delay_index = np.argmax(cross_corr[mid_index-delta:mid_index+delta]) + mid_index - delta

    # Calculate the delay in seconds
    samples_delay = (delay_index - len(signal1) + 1) - int((curr_time - sync_time) * DELAY_PER_SEC)
    time_delay = samples_delay / fs

    # print("signal1.shape: ", signal1.shape)
    # print("signal2.shape: ", signal2.shape)
    # print("correlation.shape: ", cross_corr.shape)
    # plt.figure()
    # plt.plot(cross_corr)
    # plt.grid(True)
    # plt.xlim([mid_index-1000, mid_index+1000])
    # plt.title(f'Time = {sec//60}:{sec%60}')
    # plt.show()

    return time_delay


def compute_angle(time_delay):
    arg = (SPEED_OF_SOUND * time_delay) / DIST
    rad_angle = np.arcsin(arg)
    deg_angle = np.rad2deg(rad_angle)
    return deg_angle


def localize_horn(signal1, signal2, fs, sync_time, curr_time):
    filtered_signal1 = high_pass_filter(signal1, fs, HIGH_PASS_CUTOFF)
    filtered_signal2 = high_pass_filter(signal2, fs, HIGH_PASS_CUTOFF)
    time_delay = compute_delay(filtered_signal1, filtered_signal2, fs, sync_time, curr_time)
    angle = compute_angle(time_delay)
    return angle
