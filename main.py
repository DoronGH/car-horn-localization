
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from horn_detector import *


def read_mp3_folder_np(folder_path):
    try:
        # Initialize a list to store data and sample rate for each file
        mp3_data_list = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            if filename.endswith(".mp3"):

                # Load the MP3 file
                audio = AudioSegment.from_file(file_path, format="mp3")

                # Extract the raw audio data and sample rate
                data = np.array(audio.get_array_of_samples())
                sample_rate = audio.frame_rate

                # Append data and sample rate to the list
                mp3_data_list.append((data, sample_rate))

                print(f"{filename}: ")
                horn_detect(data, sample_rate)

        return mp3_data_list
    except Exception as e:
        print(f"Error reading MP3 files from folder: {e}")
        return None


def plot_fft(signal, sample_rate):
    # Apply FFT to the signal
    fft_result = np.fft.fft(signal)

    # Calculate the frequency values corresponding to the FFT result
    freq_values = np.fft.fftfreq(len(signal), 1 / sample_rate)

    # Shift the zero frequency component to the center
    fft_result_shifted = np.fft.fftshift(fft_result)
    freq_values_shifted = np.fft.fftshift(freq_values)

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq_values_shifted, np.abs(fft_result_shifted))
    plt.title('FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

    return [fft_result_shifted, freq_values_shifted]



# Example usage
folder_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\other sounds"
mp3_data_list = read_mp3_folder_np(folder_path)
