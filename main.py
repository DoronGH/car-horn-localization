
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from horn_detector import *


def read_mp3(file_path):

    # Load the MP3 file
    audio = AudioSegment.from_file(file_path, format="mp3")

    # Extract the raw audio data and sample rate
    data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    return [data, sample_rate]


def read_mp3_folder_np(folder_path):
    try:
        # Initialize a list to store data and sample rate for each file
        mp3_data_list = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            if filename.endswith(".mp3"):

                mp3_data_list.append(read_mp3(file_path))

        return mp3_data_list
    except Exception as e:
        print(f"Error reading MP3 files from folder: {e}")
        return None


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
    plt.title(f'FFT, {i/2}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([0, 5000])
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]

f_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recordings\Recording (2).mp3"

# Example usage
data, sample_rate = read_mp3(f_path)
det = detect_horn(data, sample_rate)
for i in range(len(det)):
    if det[i]:
        print(f"{i // 60}:{i % 60}")
