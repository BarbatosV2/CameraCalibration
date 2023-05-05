import numpy as np
import cv2

# Define the number of corners on the chessboard
corners_x = 8
corners_y = 6

# Create a 3D object point matrix
object_points = np.zeros((corners_x * corners_y, 3), np.float32)
object_points[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)

# Create arrays to store object points and image points from all images
object_points_list = []
image_points_list = []

# Define the size of the chessboard square
square_size = 0.02423  # in meters, adjust according to your chessboard

# Define the termination criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Read the video from the camera
cap = cv2.VideoCapture(0)

# Calibrate the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)

    if ret:
        object_points_list.append(object_points)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points_list.append(corners)
        cv2.drawChessboardCorners(frame, (corners_x, corners_y), corners, ret)

    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)

# Create an undistortion map
h, w = gray.shape[:2]
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), 5)

# Use the calibrated camera without distortion in a live feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Remap the frame to undistort it
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imshow('Undistorted Video', undistorted_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
