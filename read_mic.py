import pyaudio
import numpy as np
import threading


p = pyaudio.PyAudio()

print("Available audio devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"{i}. {dev['name']}")

#
# # Constants
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# CHUNK = 1024
#
# # Initialize PyAudio
# p = pyaudio.PyAudio()
#
#
# # Function to handle each microphone stream
# def handle_microphone(index, stream_info):
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     input_device_index=index,
#                     frames_per_buffer=CHUNK)
#
#     print(f"Recording from {stream_info['name']}...")
#     try:
#         while True:
#             data = stream.read(CHUNK)
#             np_data = np.frombuffer(data, dtype=np.int16)
#             rms = np.sqrt(np.mean(np_data ** 2))
#             print(f"RMS from {stream_info['name']}: {rms}")
#     except KeyboardInterrupt:
#         print(f"Finished recording from {stream_info['name']}.")
#         stream.stop_stream()
#         stream.close()
#
#
# # List of microphone indices (replace these with your microphone indices)
# mic_indices = [1, 2]  # Example indices; replace with the actual indices of your mics
#
# # Create and start a thread for each microphone
# threads = []
# for index in mic_indices:
#     stream_info = p.get_device_info_by_index(index)
#     t = threading.Thread(target=handle_microphone, args=(index, stream_info))
#     t.start()
#     threads.append(t)
#
# # Join threads to main thread to keep running
# for t in threads:
#     t.join()
#
# # Cleanup on exit
# p.terminate()


# import pyaudio
#
# p = pyaudio.PyAudio()
#
# print("Available audio devices and their properties:")
# for i in range(p.get_device_count()):
#     dev = p.get_device_info_by_index(i)
#     print(f"{i}. {dev['name']}")
#     print(f"   Host API: {p.get_host_api_info_by_index(dev['hostApi'])['name']}")
#     print(f"   Input Channels: {dev['maxInputChannels']}")
#     print(f"   Output Channels: {dev['maxOutputChannels']}")
#     print(f"   Default Sample Rate: {dev['defaultSampleRate']}")
#     # Add any other properties you're interested in
#
# p.terminate()


import pyaudio
import wave

def record_audio(mic_index, duration, file_path):
    """
    Records audio from the microphone at `mic_index` for `duration` seconds
    and saves it to `file_path`.

    Parameters:
    - mic_index: Index of the microphone to use for recording.
    - duration: Duration of the recording in seconds.
    - file_path: Path to the file where the recording will be saved.
    """
    # Audio recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=CHUNK)

    print(f"Recording for {duration} seconds...")

    frames = []

    # Record for `duration` seconds
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {file_path}")

# # Example usage
# mic_index = 4  # Adjust this to the index of your microphone
# duration = 5  # Record for 5 seconds
# file_path = rf"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output_{mic_index}.wav"  # Save the recording to 'output.wav'
# record_audio(mic_index, duration, file_path)
#


import pyaudio
import wave
import threading

def record_audio(mic_index, duration, file_path):
    """
    Records audio from the microphone at `mic_index` for `duration` seconds
    and saves it to `file_path`.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=CHUNK)

    print(f"Recording from device index {mic_index} for {duration} seconds...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    modified_file_path = f"{file_path.rstrip('.wav')}_mic_{mic_index}.wav"
    with wave.open(modified_file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording from device index {mic_index} saved to {modified_file_path}")

def start_recording(mic_indices, duration, file_path):
    threads = []
    for index in mic_indices:
        t = threading.Thread(target=record_audio, args=(index, duration, file_path))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

# Example usage
mic_indices = [1, 2, 4]  # Replace with your actual microphone indices
duration = 10  # Record for 5 seconds
file_path = rf"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\output.wav"  # Base file path for recordings
start_recording(mic_indices, duration, file_path)
