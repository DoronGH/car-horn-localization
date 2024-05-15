import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract # this is tesseract module
import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os


def detect_cars(): # img: np.array):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    img = r"C:\Users\Doron\Downloads\cars2.jpg"
    # img = r"C:\Users\Doron\Downloads\street.jpeg"
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # results = model(img)  # predict on an image
    results = model.predict(source=img, show=False)
    print(results)
    # print("resultsssss: ", results[0].boxes)
    # plt.imshow(results[0].orig_img)
    # plt.axis('off')  # Turn off the axis
    # plt.show()


    print(f"bbox: {results[0].boxes}")

    img = plt.imread(img)

    fig, ax = plt.subplots()
    ax.imshow(img)

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
            black_image[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            cropped_images.append(black_image)

    fig, axs = plt.subplots(1, len(cropped_images), figsize=(15, 5))
    for i, cropped_image in enumerate(cropped_images):
        axs[i].imshow(cropped_image)
        axs[i].axis('off')
    plt.show()


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



if __name__ == '__main__':
    img = r"C:\Users\Doron\Downloads\cars2.jpg"
    # detect_cars()
    get_license_plate_from_mask(img)