import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pytesseract
import os
import easyocr
import cv2
import torch


def detect_cars(img):
    """
    Detects cars in an image using a YOLO model.
    :param img: The image in which to detect cars.
    :return: A list of cropped images of detected cars.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n.pt")
    model.to(device)
    results = model.predict(source=img, show=False)
    bounding_boxes = results[0].boxes.cls
    cropped_images = []
    for i, cls in enumerate(bounding_boxes):
        if cls == 2:
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i])
            black_image = np.zeros_like(img)
            # black_image[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            black_image[y1:y2, x1:x2] = np.ones_like(img[y1:y2, x1:x2])
            black_image[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1, 3))
            cropped_images.append(black_image)
    for i, cropped_image in enumerate(cropped_images):
        plt.imshow(cropped_image)
        plt.axis('off')
        if False:  # len(get_plate_number(cropped_image)) > 0:
            plt.title(get_plate_number(cropped_image)[0])
        else:
            plt.title(i)
        # plt.show()
    return cropped_images


def get_license_plate_from_mask(path_for_license_plates):
    """
    Extracts the license plate number from an image of a license plate.
    :param path_for_license_plates: The path to the image of the license plate.
    """
    # specify path to the license plate images folder as shown below
    list_license_plates = []
    predicted_license_plates = []
    license_plate_file = path_for_license_plates.split("/")[-1]
    license_plate, _ = os.path.splitext(license_plate_file)
    list_license_plates.append(license_plate)
    img = cv2.imread(path_for_license_plates)
    print(f"img type: {type(img)}")
    predicted_result = pytesseract.image_to_string(img, lang='eng',
                                                   config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_license_plates.append(filter_predicted_result)


def get_plate_number(img):
    """
    Uses OCR to extract the plate number from an image of a license plate.
    :param img: The image of the license plate.
    :return: A list of detected plate numbers.
    """
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    plate_numbers = []
    for detection in result:
        # print(detection[1], detection[2])
        if detection[2] > 0.1:
            plate_numbers.append(detection[1])
    return plate_numbers


def choose_mask(masks, col, angle_error):
    """
    Chooses the best mask from a list of masks based on a specified column and angle error.
    :param masks: A list of masks.
    :param col: The specified column.
    :param angle_error: The specified angle error.
    :return: The index of the best mask, or None if no suitable mask is found.
    """
    if len(masks) > 0:
        angle_mask = np.zeros(masks[0].shape)
        angle_mask[:, max(0, col - angle_error): min(angle_mask.shape[1] - 1, col + angle_error)] = 1

        # plt.imshow(angle_mask)
        # plt.title("Angle mask")
        # plt.show()

        top_mask = -1
        top_mask_value = 0
        for i, mask in enumerate(masks):
            intersect = np.logical_and(mask, angle_mask)
            value = np.sum(intersect)
            if value > top_mask_value:
                top_mask = i
                top_mask_value = value
        return top_mask
    return None


def extract_frames(video_path, start_time, n, m, video_sync_factor=0):
    """
    Extracts frames from a video.
    :param video_path: The path to the video.
    :param start_time: The start time from which to extract frames.
    :param n: The number of frames to extract.
    :param m: The interval between extracted frames.
    :param video_sync_factor: An optional video sync factor.
    :return: A numpy array of the extracted frames.
    """
    # Parse the start time
    minutes = int(start_time[:2])
    seconds = int(start_time[2:4])
    tenth_seconds = int(start_time[4:])/10
    start_time_seconds = minutes * 60 + seconds + video_sync_factor

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to start extracting from
    start_frame = int((start_time_seconds + tenth_seconds) * fps)

    # Set the current frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    count = 0
    extracted_frames = 0

    while cap.isOpened() and extracted_frames < n:
        ret, frame = cap.read()

        if not ret:
            break

        if count % m == 0:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_frames += 1

        count += 1

    cap.release()
    return np.array(frames)


def find_vehicle(video_path, start_time, n, m, col, angle_error, video_sync_factor=0):
    """
    Finds a vehicle in a video based on specified parameters.
    :param video_path: The path to the video.
    :param start_time: The start time from which to find the vehicle.
    :param n: The number of frames to consider.
    :param m: The interval between considered frames.
    :param col: The specified column.
    :param angle_error: The specified angle error.
    :param video_sync_factor: An optional video sync factor.
    :return: A list of images of the detected vehicle.
    """
    frames = extract_frames(video_path, start_time, n, m, video_sync_factor)
    res = []
    for frame in frames:
        cars_masks = detect_cars(frame)
        mask_index = choose_mask(cars_masks, col, angle_error)
        if mask_index != -1:
            # return get_plate_number(cars[mask])
            res.append(cars_masks[mask_index] * frame)
    return res


if __name__ == '__main__':
    # Example usage
    video_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\video3.mp4"
    start_time = '050105'  # Start at 00:43 (mm:ss:tt)
    n = 3  # Number of frames to extract
    m = 10  # Extract every 5th frame

    res = find_vehicle(video_path, start_time, n, m, 1350, 100)
    for frame in res:
        plt.imshow(frame)
        plt.axis('off')
        plt.title("Suspicious vehicle")
        plt.show()
