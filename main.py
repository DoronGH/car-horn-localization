
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

from horn_localization import compute_delay, high_pass_filter, localize_horn
from horn_detector import *
from syncronizer import sync


def read_audio_file(file_path):
    # Read the audio file using pydub
    audio = AudioSegment.from_file(file_path)

    # Convert audio data to NumPy array
    data = np.array(audio.get_array_of_samples())

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


def plot_fft(signal, sample_rate, txt):
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
    plt.title(f' {txt}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([0, 5000])
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]


if __name__ == '__main__':
    file_d = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\12-03\synced_mic_d_2.wav"
    file_e = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\12-03\synced_mic_e_2.wav"

    file_d = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\27-03\Clap_1_d.m4a"
    file_e = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\27-03\clap_1_e.aac"


    signal_d, fs = read_audio_file(file_d)
    signal_e, _ = read_audio_file(file_e)

    # signal_d = signal_d[100000:200000]
    # signal_e = signal_e[100000:200000]


    from pydub import AudioSegment

    # Load your stereo AAC file
    audio_stereo = AudioSegment.from_file(file_e, format="aac")

    # Convert to mono
    signal_e = audio_stereo.set_channels(1)

    a = 1+3

    # clap_1 files

    # synchronization:
    # 00:12-00:13       -1754
    # 00:20-00:21       -1746
    # 00:26.5-00:27.5   -1745
    # 00:33-00:34       -1737
    # 00:39-00:41       -1733

    # calc delay after setting  -1745 -> signal_e = signal_e[1745:]

    # first location 69 degrees when 0 is on the right:



    # clap_2 files
    # with HPF 3000
    # synchronization:
    # 00:19-00:20       -1579   -15
    # 00:23-00:24       -1580   -13
    # 00:26.5-00:27.5   -1577    0
    # 00:31-00:32       -1575    3
    # 00:34.5-00:35.5   -1583    -4

    # calc delay after setting  -1579 -> signal_e = signal_e[1579:]

    # first location 69 degrees when 0 is on the right:



    # clap_3 files
    # with HPF 3000
    # synchronization:
    # 00:15-00:17       7286
    # 00:18.5-00:19.5   7286
    # 00:21-00:23       7288
    # 00:27-00:29       7293
    # 00:30-00:32       7295


    # calc delay after setting  7288 -> signal_d = signal_d[7288:]
    # 01:32-01:34       -41.8
    # 01:41-01:44       -40.5
    # 01:53-01:55       -36.5
    # 02:13-02:15       -39.1
    # 02:22-02:24       -36.9

    # 03:25-03:28       9
    # 03:31-03:34       13.6
    # 03:39-03:41       11.1
    # 04:14.5-04:15.5   3
    # 04:18-04:20       50
    # 04:25-04:27       -10
    # 04:30.5-04:31.5   66
    # 04:34-04:35       55
    # 04:37-04:38       62










    # first location 69 degrees when 0 is on the right:


    # signal_d = signal_d[::2]
    # signal_e = signal_e[::2]
    # signal_e = np.roll(signal_d, -7586)
    # sync(signal_e[8*fs:20*fs], signal_d[8*fs:20*fs], fs)
    # detections = detect_horn(signal_d, fs)
    # for sec, detection in enumerate(detections):
    #     if detection:
    #         print(f"{sec//60}:{sec%60}")
    #         print(localize_horn(signal_d[(sec-2)*fs:(sec+3)*fs], signal_e[(sec-2)*fs:(sec+3)*fs], fs, sec))

