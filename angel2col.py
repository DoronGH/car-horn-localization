import cv2
import numpy as np

LEN_ANGLE = 95
TOLERANCE = 20


def get_frame_shape(video_path):
    """
    Get the shape of the first frame in a video file.
    :param video_path: The path to the video file.
    :return: The shape of the first frame if successful, None otherwise.
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return None

    # Read the first frame
    ret, frame = cap.read()
    if ret:
        # Get the shape of the frame
        frame_shape = frame.shape
    else:
        print("Error reading frame")
        frame_shape = None

    # Release the video capture object
    cap.release()

    return frame_shape


def angle2col(angle, video_path):
    """
    Map an angle to a column in a video frame.
    :param angle: The angle in degrees.
    :param video_path: The path to the video file.
    :return: The column corresponding to the angle if successful, None otherwise.
    """
    if np.abs(angle) > (LEN_ANGLE / 2) + TOLERANCE:
        print("Error: Angle out of range")
        return None
    frame_shape = get_frame_shape(video_path)
    if frame_shape is not None:
        frame_width = frame_shape[1]

        # map the angles inside the tolerance range
        if angle <= -(LEN_ANGLE / 2):
            return 0
        if angle >= (LEN_ANGLE / 2):
            return frame_width - 1

        # map the angle to a column
        col = int((angle + LEN_ANGLE / 2) * frame_width / LEN_ANGLE)
        return col
