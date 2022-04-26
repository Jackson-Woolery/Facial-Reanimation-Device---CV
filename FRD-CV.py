#!/usr/bin/env python
"""Computer Vision module for Facial Reanimation Device calibration.

REQUIREMENTS AND COMPATIBILITY:
Requires install of numpy, picamera, time, math, and opencv.
Built for Raspberry Pi Camera Module v2, but works with any picamera.
This file uses essential camera calibration data from the
'CV_CameraCalibrationData.npz' file created by 'CV_CameraCalibration.py'.
The camera calibration matrices must be updated using the respective file for
any new camera used.


PURPOSE AND METHODS:
This program uses computer vision techniques and opencv to capture a stream of
images and detect Aruco markers. In Stage 1, Aruco markers should be placed on
the control (naturally functioning) side of the face, and resting expression
should be neutral. When two markers are detected in this state, the 'minimum
control' distance is calculated between them. For stage 2, expression should
be changed to the largest smile possible before pressing the space bar to enter
this stage. When the two Aruco markers are detected again, the 'maximum control'
distance is calculated. The 'control distance range', and the 'control distance
difference' are then calculated and printed. ((Final control stage and
communication with Bluetooth chip functionality are in progress.))


INSTRUCTIONS:
Before you begin: Place two included Aruco markers on the subject's face using
the provided template.

Stage 1: Subject should sit in front of camera with completely relaxed facial
expression until the program shows the minimum control distance calculated.

Stage 2: Subject should smile as largely as possible before pressing 'space' to
continue to Stage 2. Hold this expression until the program shows the maximum
control distance calculated.

Stage 3: ((Functionality not included / in progress))


OUTPUTS:
The gain and limits necessary to bind the Facial Reanimation Device output to
the naturale facial range of motion are to be sent to the Facial Reanimation
Device via Bluetooth. ((This functionality is still in progress.))
"""

__author__ = "Jackson Apollo Woolery"
__email__ = "jack.woolery.94@gmail.com"

from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import cv2 as cv
import numpy as np
import math
import serial

adf = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=.1)

# Error tolerance for calibration
tolerance = 0.1


# Image capture dimensions
WIDTH, HEIGHT = 1920, 1088
width, height = str(WIDTH), str(HEIGHT)

# Display image scaling
scale = 0.5
width_d = int(WIDTH * scale)
height_d = int(HEIGHT * scale)
dim = (width_d, height_d)

# Marker length / scaling
##MARKER_LENGTH_IN = 4.5 / 25.4 # "5mm Marker"
MARKER_LENGTH_IN = 0.3175 # BS Value
print("MARKER_LENGTH_IN = ", MARKER_LENGTH_IN)

# Import Aruco marker dictionary
arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

# Import camera calibration data
KD = np.load('CV_CameraCalibrationData.npz')
K = KD['k']
DIST = KD['dist']

# Perform sub-pixel Aruco marker corner detection for increased accuracy
def sub_pix_corner_detection(img, ids, corners):
    # Convert image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create corner subpixel detection criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                100,
                0.0001
                )

    # Perform subpixel corner detection
    for tag in ids:
        for corner in corners:
            cv.cornerSubPix(image=gray_img,
                            corners=corner,
                            winSize=(2, 2),
                            zeroZone=(-1, -1),
                            criteria=criteria
                            )

    # Return updated corners
    return corners


# Calculate 3D distance between markers
def get_dist(img, ids, corners, newCamMtx):
    # Perform sub-pixel corner detection
    corners = sub_pix_corner_detection(img, ids, corners)

    # Get rotation and translation vectors with respect to marker
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        corners,
        markerLength=MARKER_LENGTH_IN,
        cameraMatrix=newCamMtx,
        distCoeffs=0
        )

    # Extract translation vectors for each marker
    tvec0 = tvecs[0][0]
    tvec1 = tvecs[1][0]

    # Calculate distance between markers
    distance = math.sqrt((tvec0[0] - tvec1[0]) ** 2 +
                         (tvec0[1] - tvec1[1]) ** 2 +
                         (tvec0[2] - tvec1[2]) ** 2
                         )

    # Draw marker borders & axes on display image
    cv.aruco.drawDetectedMarkers(image=img,
                                 corners=corners,
                                 ids=ids,
                                 borderColor=(0, 0, 255)
                                 )

    # Resize display image
    disp_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # Return calculated distance and marked display image
    return distance, disp_img


if __name__ == '__main__':
    # Initialize Camera
    camera = PiCamera()
    camera.resolution = (WIDTH, HEIGHT)
    camera.exposure_mode = 'sports'
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

    # Initialize stage statuses
    gotMin = False
    gotMax = False
    setMin = False
    setMax = False

    # For each frame...
    for frame in camera.capture_continuous(rawCapture,
                                           format="bgr",
                                           use_video_port=True
                                           ):
        # Get image and clear stream
        img = frame.array
        rawCapture.truncate(0)

        # Get optimal camera matrix
        newCamMtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix=K,
                                                      distCoeffs=DIST,
                                                      imageSize=(WIDTH, HEIGHT),
                                                      alpha=1,
                                                      newImgSize=(WIDTH, HEIGHT)
                                                      )

# Correct image using optimal camera matrix
##        corr_img = cv.undistort(img,
##                                K,
##                                DIST,
##                                None,
##                                newCamMtx
##                                )

        # Detect Aruco marker ids and corners
        corners, ids, _ = cv.aruco.detectMarkers(image=img,
                                                 dictionary=arucoDict,
                                                 cameraMatrix=K,
                                                 distCoeff=DIST
                                                 )

        # Initialize detected marker count
        count = 0

        # Wait to hit target minimum (Stage 4)
        if setMax == True:
            if setMin == False:
                print("STAGE 4: SET MINIMUM")
                if ids is not None:
                    for tag in ids:
                        if tag == 2:
                            count = count + 1
                            print("Aruco 2 Detected")
                        if tag == 3:
                            count = count + 1
                            print("Aruco 3 Detected")

                if count >= 2:
                    currDist, maxSetImg = get_dist(img, ids, corners, newCamMtx)
                    if currDist < minDist + tolerance * minDist:
                        setMax = True
                        cv.imshow("Max Set Img", maxSetImg)
                        cv.waitKey(0)

        # Wait to hit target maximum (Stage 3)
        if gotMin:
            if not setMax:
                if ids is not None:
                    print("STAGE 3: SET MAXIMUM")
                    for tag in ids:
                        if tag == 2:
                            count = count + 1
                            print("Aruco 2 Detected")
                        if tag == 3:
                            count = count + 1
                            print("Aruco 3 Detected")

                if count >= 2:
                    currDist, maxSetImg = get_dist(img, ids, corners, newCamMtx)
                    if currDist >= maxDist:
                        setMax = True
                        cv.imshow("Max Set Img", maxSetImg)
                        cv.waitKey(0)

        # Get maximum control distance (Stage 2)
        if gotMax:
            if not gotMin:
                if ids is not None:
                    print("STAGE 2: GET MINIMUM")
                    for tag in ids:
                        if tag == 0:
                            count = count + 1
                            print("Aruco 0 Detected")
                        if tag == 1:
                            count = count + 1
                            print("Aruco 1 Detected")

                # When two markers are detected, get distance and show image                        
                if count >= 2:
                    minDist, minImg = get_dist(img, ids, corners, newCamMtx)
                    print("Minimum Distance: ", minDist)
                    gotMax = True

                    cv.imshow("Min Control Img", minImg)
                    cv.waitKey(0)

                    # Print calibration distance range and difference
                    print("Control Distance Range: ",
                          round(minDist, 4), " - ",
                          round(maxDist, 4), " inches")
                    print("Control Distance Difference : ",
                          round(maxDist - minDist, 4), " inches")


        # Get maximum control distance (Stage 1)
        if not gotMax:
            if ids is not None:
                print("STAGE 1: GET MAXIMUM")
                for tag in ids:
                    if tag == 0:
                        count = count + 1
                        print("Aruco 0 Detected")
                    if tag == 1:
                        count = count + 1
                        print("Aruco 1 Detected")

            # When two markers are detected, get distance and show image        
            if count >= 2:
                maxDist, maxImg = get_dist(img, ids, corners, newCamMtx)
                print("Maximum Distance: ", maxDist)
                gotMax = True

                cv.imshow("Max Control Img", maxImg)
                cv.waitKey(0)


        # Resize and show live stream feed
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

        cv.imshow("Stream", resized)
        cv.waitKey(1)

        rawCapture.truncate(0)

# Debugging: Show detected markers
##        if ids is not None:
##            det_img = cv.aruco.drawDetectedMarkers(image=img,
##                                                   corners=corners,
##                                                   ids=ids,
##                                                   borderColor=(0, 0, 255)
##                                                   )
##            
##            det_img = cv.resize(det_img, dim, interpolation=cv.INTER_AREA)
##
##            cv.imshow("Detected Markers", det_img)
##            cv.waitKey(1)
