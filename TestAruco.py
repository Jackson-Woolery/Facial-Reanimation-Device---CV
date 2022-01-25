from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import cv2 as cv
import numpy as np
import math

WIDTH, HEIGHT = 1920, 1080
width, height = str(WIDTH), str(HEIGHT)

scale = 1/4
width2 = int(WIDTH * scale)
height2 = int(HEIGHT * scale)
dim = (width2, height2)

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

KD = np.load('CV_CameraCalibrationData.npz')
K = KD['k']
DIST = KD['dist']

camera = PiCamera()
camera.resolution = (WIDTH, HEIGHT)
camera.exposure_mode = 'sports'
rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array

    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    
    cv.imshow("mainstream", resized)
    cv.waitKey(1)
    rawCapture.truncate(0)

    corners, ids, _ = cv.aruco.detectMarkers(image=img, dictionary=arucoDict, cameraMatrix=K, distCoeff=DIST)

    if ids is None:
        print("Marker not detected.")

    if ids is not None:
        for tag in ids:
            cv.aruco.drawDetectedMarkers(image=img, corners=corners, ids=ids, borderColor=(0, 0, 255))
            resized2 = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            cv.imshow("Stream - DETECTED", resized2)
            cv.waitKey(1)
