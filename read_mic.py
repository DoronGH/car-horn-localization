import pyaudio
import numpy as np
import threading


# p = pyaudio.PyAudio()
#
# print("Available audio devices:")
# for i in range(p.get_device_count()):
#     dev = p.get_device_info_by_index(i)
#     print(f"{i}. {dev['name']}")


# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()


# Function to handle each microphone stream
def handle_microphone(index, stream_info):
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=index,
                    frames_per_buffer=CHUNK)

    print(f"Recording from {stream_info['name']}...")
    try:
        while True:
            data = stream.read(CHUNK)
            np_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np_data ** 2))
            print(f"RMS from {stream_info['name']}: {rms}")
    except KeyboardInterrupt:
        print(f"Finished recording from {stream_info['name']}.")
        stream.stop_stream()
        stream.close()


# List of microphone indices (replace these with your microphone indices)
mic_indices = [1, 2]  # Example indices; replace with the actual indices of your mics

# Create and start a thread for each microphone
threads = []
for index in mic_indices:
    stream_info = p.get_device_info_by_index(index)
    t = threading.Thread(target=handle_microphone, args=(index, stream_info))
    t.start()
    threads.append(t)

# Join threads to main thread to keep running
for t in threads:
    t.join()

# Cleanup on exit
p.terminate()
