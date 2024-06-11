import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy.signal import correlate, filtfilt, butter
from horn_localization import HIGH_PASS_CUTOFF

CORR_LEN = 1


def read_audio_file(file_path):
    # Read the audio file using pydub
    audio = AudioSegment.from_file(file_path)

    audio_mono = audio.set_channels(1)

    # Convert audio data to NumPy array
    data = np.array(audio_mono.get_array_of_samples())

    # Get the sample rate
    sample_rate = audio.frame_rate

    return data, sample_rate


def save_as_wav(audio_data, sample_rate, output_file):
    # Create an AudioSegment from the NumPy array
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1  # assuming a mono signal, adjust if needed
    )

    # Save the AudioSegment to a WAV file
    audio.export(output_file, format="wav")


def high_pass_filter(signal, sample_rate, cutoff):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist

    # Design high-pass Butterworth filter
    b, a = butter(N=4, Wn=normal_cutoff, btype='high', analog=False)

    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


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
    plt.figure()
    plt.plot(cross_corr)
    plt.grid(True)
    plt.xlim([delay_index - 1000, delay_index + 1000])
    plt.title(f'index = {i}')
    plt.show()

    return samples_delay


def sync(signal1, signal2, fs):
    filtered_signal1 = high_pass_filter(signal1, fs, HIGH_PASS_CUTOFF)
    filtered_signal2 = high_pass_filter(signal2, fs, HIGH_PASS_CUTOFF)
    time_len = len(signal1) // fs
    range_size = (time_len - CORR_LEN) * 2 + 1
    for n in range(range_size):
        print(f"index: {n}")
        print(compute_delay(filtered_signal1[int(0.5 * n) * fs:int(CORR_LEN + 0.5 * n) * fs],
                            filtered_signal2[int(0.5 * n) * fs:int(CORR_LEN + 0.5 * n) * fs], n))


if __name__ == '__main__':
    file_d = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\with_video_2_d.m4a"
    file_e = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\with_video_2_e.aac"
    signal_d, fs = read_audio_file(file_d)
    signal_e, _ = read_audio_file(file_e)
    signal_e = signal_e[2530:]
    sync_d = signal_d[32*fs:62*fs]
    sync_e = signal_e[32*fs:62*fs]
    sync(sync_d, sync_e, fs)
    # save_as_wav(signal_d, fs, "synced_with_video_2_d.wav")
    # save_as_wav(signal_e, fs, "synced_with_video_2_e.wav")
