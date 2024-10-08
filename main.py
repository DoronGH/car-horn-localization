from matplotlib import pyplot as plt
from pydub import AudioSegment
from angel2col import angle2col
from horn_localization import localize_horn
from horn_detector import *
from visual_detection import find_vehicle

MIN_DETECTIONS = 5
NUM_OF_FRAMES = 3
FRAME_DIFF = 10
PIXEL_TOLERANCE = 50
SYNC_TIME = 40

AUDIO1_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\synced_with_video_2_d_40.wav"
AUDIO2_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\synced_with_video_2_e_40.wav"
VIDEO_PATH = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\video2.mp4"


def read_audio_file(file_path):
    """
    Reads an audio file and returns the data and sample rate.
    :param file_path: The path to the audio file.
    :return: A tuple containing the audio data as a NumPy array and the sample rate.
    """
    # Read the audio file using pydub
    audio = AudioSegment.from_file(file_path)

    audio_mono = audio.set_channels(1)

    # Convert audio data to NumPy array
    data = np.array(audio_mono.get_array_of_samples())

    # Get the sample rate
    sample_rate = audio.frame_rate

    return data, sample_rate


def print_detection(sec):
    """
    Prints the detection time in a specific format.
    :param sec: The detection time in seconds.
    """
    if sec // 60 >= 10 and sec % 60 >= 10:
        print(f"Horn Detected! - {int(sec // 60)}:{sec % 60}")
    elif sec // 60 < 10 and sec % 60 >= 10:
        print(f"Horn Detected! - 0{int(sec // 60)}:{sec % 60}")
    elif sec // 60 >= 10 and sec % 60 < 10:
        print(f"Horn Detected! - {int(sec // 60)}:0{sec % 60}")
    else:
        print(f"Horn Detected! - 0{int(sec // 60)}:0{sec % 60}")


def enough_detections(detections):
    """
    Checks if the number of detections is greater than or equal to the minimum required.
    :param detections: A list of boolean values indicating the presence of a horn-like sound in each sub-array.
    :return: True if the number of detections is sufficient, False otherwise.
    """
    return np.sum(detections) >= MIN_DETECTIONS


def find_angle(audio1, audio2, fs, detections, start):
    """
    Finds the median angle of arrival of a horn sound based on multiple detections.
    :param audio1: The first audio signal.
    :param audio2: The second audio signal.
    :param fs: The sample rate of the audio signals.
    :param detections: A list of boolean values indicating the presence of a horn-like sound in each sub-array.
    :param start: The start time of the detections.
    :return: The median angle of arrival of the horn sound in degrees.
    """
    angles = []
    for half_sec, detection in enumerate(detections):
        if detection:
            sec = start + (half_sec / 2)
            signal1 = audio1[int((sec-0.5)*fs):int((sec+1)*fs)]
            signal2 = audio2[int((sec-0.5)*fs):int((sec+1)*fs)]
            angle = localize_horn(signal1, signal2, fs, SYNC_TIME, sec)
            if not np.isnan(angle):
                angles.append(angle)
    return np.median(angles)


def time_format(sec):
    """
    Formats a time value in seconds to a specific string format.
    :param sec: The time value in seconds.
    :return: The formatted time string.
    """
    if sec // 60 >= 10 and sec % 60 >= 10:
        return f"{int(sec // 60)}{int(sec % 60)}{int(sec * 100) % 100}"
    elif sec // 60 < 10 and sec % 60 >= 10:
        return f"0{int(sec // 60)}{int(sec % 60)}{int(sec * 100) % 100}"
    elif sec // 60 >= 10 and sec % 60 < 10:
        return f"{int(sec // 60)}0{int(sec % 60)}{int(sec * 100) % 100}"
    else:
        return f"0{int(sec // 60)}0{int(sec % 60)}{int(sec * 100) % 100}"


def plot_results(results, start_time):
    """
    Plots the results of the vehicle detection.
    :param results: The results of the vehicle detection.
    :param start_time: The start time of the detections.
    """
    for frame in results:
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Vehicle Detected\nTime: {start_time[:2]}:{start_time[2:4]}.{start_time[4]}")
        plt.show()


def main():
    """
    The main function of the script. It reads the audio files, detects the horn sounds, localizes the horn, and plots
    the results.
    """
    audio1, fs = read_audio_file(AUDIO1_PATH)
    audio2, _ = read_audio_file(AUDIO2_PATH)
    print("Audio files read successfully")
    print("Searching for horn...")
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
                    plot_results(results, start_time)


if __name__ == '__main__':
    main()
