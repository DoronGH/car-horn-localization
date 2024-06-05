import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract # this is tesseract module
import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os
import easyocr
from PIL import Image


def detect_cars(img): # img: np.array):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # img = r"C:\Users\Doron\Downloads\street.jpeg"
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # results = model(img)  # predict on an image
    results = model.predict(source=img, show=False)
    # print("resultsssss: ", results[0].boxes)
    # plt.imshow(results[0].orig_img)
    # plt.axis('off')  # Turn off the axis
    # plt.show()


    # img = plt.imread(img)
    #
    # fig, ax = plt.subplots()
    # ax.imshow(img)

    bounding_boxes = results[0].boxes.cls

    # for i, cls in enumerate(bounding_boxes):
    #     if cls == 2:
    #         x1, y1, x2, y2 = results[0].boxes.xyxy[i]
    #         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    #         ax.add_patch(rect)
    #
    # # Show the plot
    # plt.show()

    cropped_images = []

    for i, cls in enumerate(bounding_boxes):
        if cls == 2:
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i])
            black_image = np.zeros_like(img)
            # black_image[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            black_image[y1:y2, x1:x2] = np.ones_like(img[y1:y2, x1:x2])
            black_image[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1,3))
            cropped_images.append(black_image*255)

    # fig, axs = plt.subplots(1, len(cropped_images), figsize=(15, 5))
    # for i, cropped_image in enumerate(cropped_images):
    #     axs[i].imshow(cropped_image)
    #     axs[i].axis('off')
    # plt.show()
    for i, cropped_image in enumerate(cropped_images):
        plt.imshow(cropped_image)
        plt.axis('off')
        if len(get_plate_number(cropped_image)) > 0:
            plt.title(get_plate_number(cropped_image)[0])
        else:
            plt.title(i)
        plt.show()
    return cropped_images


def get_license_plate_from_mask(path_for_license_plates):
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
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    plate_numbers= []
    for detection in result:
        # print(detection[1], detection[2])
        if detection[2] > 0.1:
            plate_numbers.append(detection[1])
    return plate_numbers


def choose_mask(masks, col, angle_error):
    if len(masks) > 0:
        angle_mask = np.zeros(masks[0].shape)
        angle_mask[:, col - angle_error: col + angle_error] = 1

        plt.imshow(angle_mask)
        plt.title("Angle mask")
        plt.show()

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


import cv2
import matplotlib.pyplot as plt


def extract_frames(video_path, start_time, n, m):
    # Parse the start time
    minutes = int(start_time[:2])
    seconds = int(start_time[2:])
    start_time_seconds = minutes * 60 + seconds

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to start extracting from
    start_frame = int(start_time_seconds * fps)

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


if __name__ == '__main__':
    img = r"C:\Users\Doron\Downloads\cars2.jpg"
    # pil_img = Image.open(img)

    # img_array = np.array(pil_img)

    # Example usage
    video_path = r"G:\.shortcut-targets-by-id\1WhfQEk4yh3JFs8tCyjw2UuCdUSe6eKzw\Engineering project\with_video\02-06\video3.mp4"
    start_time = '0043'  # Start at 01:30 (mm:ss)
    n = 1  # Number of frames to extract
    m = 5  # Extract every 5th frame

    frames = extract_frames(video_path, start_time, n, m)

    # Display the extracted frames using matplotlib

    for i, frame in enumerate(frames):
        plt.imshow(frame)
        plt.title(f'Frame {i}')
        plt.axis('off')
        plt.show()

    cars = detect_cars(frame)
    for car_mask in cars:
        masked = frames * car_mask
        plt.imshow(masked[0])
        plt.title(f'Frame {i} masked')
        plt.axis('off')
        plt.show()


    # print(f"img shape: {cars[0].shape}")
    # det = choose_mask(cars, 200, 20)
    # print(f"det: {det}")
    # for i, car in enumerate(cars):
    #     print(f"car {i}, license plate: {get_plate_number(car)}")





