
import threading
import pyaudio
import wave


def record_audio(mic_index, duration, file_path):
    """
    Records audio from the microphone at `mic_index` for `duration` seconds
    and saves it to `file_path`.
    :param mic_index: The index of the microphone.
    :param duration: The duration of the recording in seconds.
    :param file_path: The path to the file where the recording will be saved.
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
    """
    Starts recording audio from the microphones at `mic_indices` for `duration` seconds
    and saves it to `file_path`.
    :param mic_indices: A list of indices of the microphones.
    :param duration: The duration of the recording in seconds.
    :param file_path: The path to the file where the recording will be saved.
    """
    threads = []
    for index in mic_indices:
        t = threading.Thread(target=record_audio, args=(index, duration, file_path))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


# Example usage
if __name__ == '__main__':
    mic_indices = [1, 9]  # Replace with your actual microphone indices
    duration = 5  # Record for 5 seconds
    file_path = rf"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\recording attempts\soud_card_test.wav"  # Base file path for recordings
    start_recording(mic_indices, duration, file_path)
