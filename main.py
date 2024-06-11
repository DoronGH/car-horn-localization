
import os
import sys
from pydub import AudioSegment
from angel2col import angle2col
from horn_localization import localize_horn
from horn_detector import *
from visual_detection import find_vehicle

MIN_DETECTIONS = 5
NUM_OF_FRAMES = 3
FRAME_DIFF = 10
PIXEL_TOLERANCE = 200

AUDIO1_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\synced_with_video_2_d.wav"
AUDIO2_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\synced_with_video_2_e.wav"
VIDEO_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\video2.mp4"


def read_audio_file(file_path):
    # Read the audio file using pydub
    audio = AudioSegment.from_file(file_path)

    audio_mono = audio.set_channels(1)

    # Convert audio data to NumPy array
    data = np.array(audio_mono.get_array_of_samples())

    # Get the sample rate
    sample_rate = audio.frame_rate

    return data, sample_rate


def print_detection(sec):
    if sec // 60 >= 10 and sec % 60 >= 10:
        print(f"{int(sec // 60)}:{int(sec % 60)}")
    elif sec // 60 < 10 and sec % 60 >= 10:
        print(f"0{int(sec // 60)}:{int(sec % 60)}")
    elif sec // 60 >= 10 and sec % 60 < 10:
        print(f"{int(sec // 60)}:0{int(sec % 60)}")
    else:
        print(f"0{int(sec // 60)}:0{int(sec % 60)}")


def enough_detections(detections):
    return np.sum(detections) >= MIN_DETECTIONS


def find_angle(audio1, audio2, fs, detections, start):
    angles = []
    for half_sec, detection in enumerate(detections):
        if detection:
            sec = start + (half_sec / 2)
            signal1 = audio1[int((sec-0.5)*fs):int((sec+1)*fs)]
            signal2 = audio2[int((sec-0.5)*fs):int((sec+1)*fs)]
            angle = localize_horn(signal1, signal2, fs)
            angles.append(angle)
    return np.median(angles)


def time_format(sec):
    if sec // 60 >= 10 and sec % 60 >= 10:
        return f"{int(sec // 60)}{int(sec % 60)}"
    elif sec // 60 < 10 and sec % 60 >= 10:
        return f"0{int(sec // 60)}{int(sec % 60)}"
    elif sec // 60 >= 10 and sec % 60 < 10:
        return f"{int(sec // 60)}0{int(sec % 60)}"
    else:
        return f"0{int(sec // 60)}0{int(sec % 60)}"


def plot_results(results):
    for frame in results:
        plt.imshow(frame)
        plt.axis('off')
        plt.title("Vehicle detection")
        plt.show()


def main():
    audio1, fs = read_audio_file(AUDIO1_PATH)
    audio2, _ = read_audio_file(AUDIO2_PATH)
    detections = detect_horn(audio1, fs)
    for half_sec, detection in enumerate(detections):
        if detection:
            sec = half_sec / 2
            print_detection(sec)
            end = min(len(detections)-1, half_sec+20)
            if enough_detections(detections[half_sec:end]):
                print("Enough detections, localizing horn...")
                angle = find_angle(audio1, audio2, fs, detections[half_sec:end], sec)
                print(f"Angle: {angle}")
                col = angle2col(angle, VIDEO_PATH)
                if col is not None:
                    start_time = time_format(sec)
                    results = find_vehicle(VIDEO_PATH, start_time, NUM_OF_FRAMES, FRAME_DIFF, col, PIXEL_TOLERANCE)
                    plot_results(results)


if __name__ == '__main__':
    main()


    # if len(sys.argv) != 4:
    #     print("Usage: python main.py <audio1_path> <audio2_path> <video_path>")
    #     sys.exit(1)
    # audio1_path = sys.argv[1]
    # audio2_path = sys.argv[2]
    # video_path = sys.argv[3]