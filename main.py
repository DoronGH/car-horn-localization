
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

from horn_localization import compute_delay, high_pass_filter, localize_horn
from horn_detector import *


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
    file_d = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\synced_analog_mic_1_d.wav"
    file_e = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\Delayed Recoding\synced_analog_mic_1_e.wav"
    data_d, fs = read_audio_file(file_d)
    data_e, _ = read_audio_file(file_e)
    detections = detect_horn(data_d, fs)
    for sec, detection in enumerate(detections):
        if detection:
            signal_d = high_pass_filter(data_d[int(sec-2)*fs:int(sec+3)*fs], fs, cutoff=500)
            signal_e = high_pass_filter(data_e[int(sec-2)*fs:int(sec+3)*fs], fs, cutoff=500)
            print(f"{sec//60}:{sec%60}")
            print(localize_horn(signal_d, signal_e, fs, sec))


