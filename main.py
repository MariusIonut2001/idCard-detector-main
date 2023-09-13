import cv2
import numpy as np
import dlib
from imutils import face_utils
from utils import for_point_warp
from datetime import datetime

# Load the face detection model and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\mariu\\Desktop\\shape_predictor_68_face_landmarks.dat')

# Capture video from the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize a flag to indicate if a card is found
card_found = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([80, 0, 139])
    upper_blue = np.array([111, 118, 220])

    # Create a mask for the blue color
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Apply the mask to the original frame
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Define color ranges for different regions
    color_ranges = [
        ([80, 0, 139], [111, 118, 220]),
        ([6, 79, 112], [29, 255, 220]),
        ([105, 68, 90], [180, 239, 170]),
        ([0, 44, 100], [7, 200, 200])
    ]

    # Initialize an array to store the values for each color range
    values = [0] * len(color_ranges)

    # Iterate through color ranges and calculate values
    for i, (lower_color, upper_color) in enumerate(color_ranges):
        color_mask = cv2.inRange(hsv_frame, np.array(lower_color), np.array(upper_color))
        values[i] = np.sum(color_mask == 255)

    # Calculate the result based on color values
    result = values[0] - sum(values[1:])

    font = cv2.FONT_HERSHEY_SIMPLEX

    if result >= 3000:
        # Perform image processing on the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        _, threshold_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
        edge_frame = cv2.Canny(gray_frame, 50, 150)

        # Combine thresholded and edge frames
        combined_frame = threshold_frame + edge_frame

        # Perform dilation
        kernel = np.ones((5, 5), np.uint8)
        dilation_frame = cv2.dilate(combined_frame, kernel)

        # Find contours in the dilation frame
        contours, _ = cv2.findContours(dilation_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with maximum area
        max_area = 0
        max_contour = None

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            # Draw the bounding box around the maximum contour
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # Add the "Card Found" text in green
            cv2.putText(frame, 'Card Found', (30, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Perform further processing on the detected region
            warped = for_point_warp(box, frame)

            # Display the perspective-transformed image in a new window
            cv2.imshow('Warped Image', warped)

            # Set the flag to indicate that a card is found
            card_found = True

    else:
        cv2.putText(frame, 'This card is not an ID Card', (30, 50), font, 0.8, (100, 100, 255), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Result', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
