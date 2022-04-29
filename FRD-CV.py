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
The purpose of this program is to demonstrate the capability of a Computer
Vision module to be used to calibrate the Facial Reanimation Device, as
designed by the Mines Neuromuscular Mediators. While the final program would
ideally have wireless communication with the implanted device, and calibrate
it to create symmetrical maximum and minimum expressions - this program
functions as a demonstration and proof of concept that it can successfully
detect the maximum and minimum required, and quickly and accurately detect
and react when the opposite side of the face has reached these same calibration
points by saving the matching images and distances.

Methods used include computer vision techniques and opencv to capture a stream
of images and detect Aruco markers. Three-dimensional data is gathered from
these markers using openCV, and distances are calculated between them using a
function which performs trigonometric functions on the translation vectors of
the Aruco markers detected.

Stages 1 and 2 wait until the correct markers (0 and 1) are detected, and the
maximum and minimum expressive "control" distances are calculated using the
aforementioned function. A still image is captured in each stage when the
correct markers are detected and the measurements are acquired.

Stages 3 and 4 look for markers 2 and 3 on the "actuated" side of the face. In
theory, these are the stages that would run a wireless feedback control system
with the implanted device - instructing it to actuate and de-actuate until each
of the control measurements are matched, at which point this program would flag
the implanted device to save the current required for each as a maximum and
minimum. However, for practical purposes of this demonstration program, a person
with no facial paralysis emulates this actuation and de-actuation by smiling,
and then relaxing their face in stages 3 and 4 respectively. Live measurements
are displayed on the stream in these stages, along with the respective control
measurements so users can see the target distance, and the current distance to
compare. These measurements are also displayed on the saved stills taken at the
time of match detection for accuracy comparison.


INSTRUCTIONS:
Before you begin: Place 5mm Aruco marker 0 just below the corner of the lip on
the "control" or healthy side of the face, and Aruco 1 at the peak of the
zygomatic arch on the same side. Similarly, place Aruco markers 2 and 3 at the
same locations respectively on the opposite side of the face which will be
referenced as the "actuated" side of the face.

Stage 1: Subject should sit in front of camera with the largest smile possible
until the program shows the maximum control distance calculated, and captures
an image. Ensure that Aruco markers 0 and 1 are within unobstructed view of the
PiCamera.

Stage 2: Subject should relax their face as possible before pressing 'space' to
continue to Stage 2. Hold this expression until the program shows the minimum
control distance calculated. Ensure that Aruco markers 0 and 1 are within
unobstructed view of the PiCamera.

Stage 3: Press 'space'. Subject should smile slowly on the "actuated" side until
the program detects the same expressive distance as the control side and
captures a still image. Current and control distances will be shown on the
stream, meanwhile. Ensure that Aruco markers 2 and 3 are within unobstructed
view of the PiCamera.

Stage 4: Press 'space'. Subject should slowly relax their face on the "actuated"
side until the program detects the same expressive distance as the control side
and captures a still image. Current and control distances will be displayed on
the stream, meanwhile. Ensure that Aruco markers 2 and 3 are within unobstructed
view of the PiCamera.


OUTPUTS:
This program outputs images capturing the control and "actuated" maximum and
minimum expressions. It also prints the expressive distance calculated for
each image saved.
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

# Image capture dimensions
WIDTH, HEIGHT = 1920, 1088
width, height = str(WIDTH), str(HEIGHT)

# Display image scaling
scale = 0.5
width_d = int(WIDTH * scale)
height_d = int(HEIGHT * scale)
dim = (width_d, height_d)

# Marker length / scaling
MARKER_LENGTH_IN = 0.3175  # BS Value

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

    print("STAGE 1: GET MAXIMUM")

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

        # Detect Aruco marker ids and corners
        corners, ids, _ = cv.aruco.detectMarkers(image=img,
                                                 dictionary=arucoDict,
                                                 cameraMatrix=K,
                                                 distCoeff=DIST
                                                 )

        # Initialize and clear counters
        count = 0
        i = 0
        j = 0
        keepIDs = []
        keepCorners = []

        # Wait to hit target minimum (Stage 4)
        if setMax:
            if not setMin:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        # Search for correct marker IDs and save only the correct IDs and corners
                        # Counter is incremented in order to tell when both correct markers are detected
                        if tag == 2:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner2 in corners:
                                if j == i:
                                    keepCorners.append(corner2)
                                j = j + 1
                        if tag == 3:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner3 in corners:
                                if j == i:
                                    keepCorners.append(corner3)
                                j = j + 1
                        i = i + 1

                # If both correct markers are detected...
                if count >= 2:
                    # Calculate and display current distance on the stream in red
                    currDist, minSetImg = get_dist(img,
                                                   keepIDs,
                                                   keepCorners,
                                                   newCamMtx
                                                   )
                    strDist = str(currDist)
                    cv.putText(img,
                               "Current Distance: " + strDist,
                               (5, 50),
                               cv.FONT_HERSHEY_SIMPLEX,
                               2,
                               (0, 0, 255),
                               2
                               )
                    # Display the Minimum Control Distance on the stream in green
                    strMin = str(minDist)
                    cv.putText(img,
                               "Control Minimum: " + strMin,
                               (5, 150),
                               cv.FONT_HERSHEY_SIMPLEX,
                               2,
                               (0, 255, 0),
                               2
                               )

                    # If the current distance has reached the (minimum) calibration point...
                    if currDist >= minDist:
                        setMin = True
                        # Print the "Actuated" maximum distanced saved, and display it on the saved image in red
                        print("Actuated Min Distance: ", currDist)
                        cv.putText(minSetImg,
                                   "Actuated Min Distance: " + strDist,
                                   (5, 60),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 0, 255),
                                   2
                                   )
                        # Display the minimum control distance on the saved image in green
                        cv.putText(minSetImg,
                                   "Control Min Distance: " + strMin,
                                   (5, 30),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 255, 0),
                                   2
                                   )
                        # Show the Actuated Minimum Image at the bottom right of the display
                        cv.imshow("Actuated Min Img", minSetImg)
                        cv.moveWindow("Actuated Min Img", 1080, 720)
                        cv.waitKey(0)
                        # Calibration is complete
                        print("CALIBRATION COMPLETE!")

        # Wait to hit target maximum (Stage 3)
        if gotMin:
            if not setMax:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        # Search for correct marker IDs and save only the correct IDs and corners
                        # Counter is incremented in order to tell when both correct markers are detected
                        if tag == 2:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner2 in corners:
                                if j == i:
                                    keepCorners.append(corner2)
                                j = j + 1
                        if tag == 3:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner3 in corners:
                                if j == i:
                                    keepCorners.append(corner3)
                                j = j + 1
                        i = i + 1

                # If both correct markers are detected...
                if count >= 2:
                    # Calculate and display current distance on the stream in red
                    currDist, maxSetImg = get_dist(img,
                                                   keepIDs,
                                                   keepCorners,
                                                   newCamMtx
                                                   )
                    strDist = str(currDist)
                    cv.putText(img,
                               "Current Distance: " + strDist,
                               (5, 100),
                               cv.FONT_HERSHEY_SIMPLEX,
                               1.5,
                               (0, 0, 255),
                               2
                               )
                    # Display the Maximum Control Distance on the stream in green
                    strMax = str(maxDist)
                    cv.putText(img,
                               "Control Maximum: " + strMax,
                               (5, 50),
                               cv.FONT_HERSHEY_SIMPLEX,
                               1.5,
                               (0, 255, 0),
                               2
                               )

                    # If the current distance has reached the calibration (maximum) point...
                    if currDist <= maxDist:
                        setMax = True
                        # Print the "Actuated" maximum distanced saved, and display it on the saved image in red
                        print("Actuated Max Distance: ", currDist)
                        strDist = str(currDist)
                        cv.putText(maxSetImg,
                                   "Actuated Max Distance: " + strDist,
                                   (5, 60),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 0, 255),
                                   2
                                   )
                        strMax = str(maxDist)
                        # Display the maximum control distance on the saved image in green
                        cv.putText(maxSetImg,
                                   "Control Max Distance: " + strMax,
                                   (5, 30),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 255, 0),
                                   2
                                   )
                        # Show the Actuated Maximum Image at the top right of the display
                        cv.imshow("Actuated Max Img", maxSetImg)
                        cv.moveWindow("Actuated Max Img", 1080, 52)
                        cv.waitKey(0)
                        # Proceed to Stage 4
                        print("STAGE 4: SET MINIMUM")

        # Get maximum control distance (Stage 2)
        if gotMax:
            if not gotMin:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        # Search for correct marker IDs and save only the correct IDs and corners
                        # Counter is incremented in order to tell when both correct markers are detected
                        if tag == 0:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner0 in corners:
                                if j == i:
                                    keepCorners.append(corner0)
                                j = j + 1
                        if tag == 1:
                            count = count + 1
                            keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                            for corner1 in corners:
                                if j == i:
                                    keepCorners.append(corner1)
                                j = j + 1
                        i = i + 1

                # If both correct markers are detected...
                if count >= 2:
                    # Calculate and print distance
                    minDist, minImg = get_dist(img,
                                               keepIDs,
                                               keepCorners,
                                               newCamMtx
                                               )
                    print("Minimum Distance: ", minDist)
                    gotMin = True
                    # Show Minimum Control Image at the bottom left of the display
                    cv.imshow("Min Control Img", minImg)
                    cv.moveWindow("Min Control Img", 0, 720)
                    cv.waitKey(0)

                    # Print calibration distance range and difference
                    print("Control Distance Range: ",
                          round(minDist, 4), " - ",
                          round(maxDist, 4), " inches")
                    print("Control Distance Difference : ",
                          round(maxDist - minDist, 4), " inches")

                    # Proceed to Stage 3
                    print("STAGE 3: SET MAXIMUM")

        # Get maximum control distance (Stage 1)
        if not gotMax:
            if ids is not None:
                keepCorners = []
                for tag in ids:
                    j = 0
                    # Search for correct marker IDs and save only the correct IDs and corners
                    # Counter is incremented in order to tell when both correct markers are detected
                    if tag == 0:
                        count = count + 1
                        keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                        for corner0 in corners:
                            if j == i:
                                keepCorners.append(corner0)
                            j = j + 1
                    if tag == 1:
                        count = count + 1
                        keepIDs = np.append(keepIDs, [tag[0]], axis=0)
                        for corner1 in corners:
                            if j == i:
                                keepCorners.append(corner1)
                            j = j + 1
                    i = i + 1

            # If both correct markers are detected...
            if count >= 2:
                # Calculate and print distance
                maxDist, maxImg = get_dist(img,
                                           keepIDs,
                                           keepCorners,
                                           newCamMtx
                                           )
                print("Maximum Distance: ", maxDist)
                gotMax = True
                # Show Maximum Control Image at the top left of the display
                cv.imshow("Max Control Img", maxImg)
                cv.moveWindow("Max Control Img", 0, 52)
                cv.waitKey(0)
                # Proceed to Stage 2
                print("STAGE 2: GET MINIMUM")

        # Resize and show live stream feed
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        cv.imshow("Stream", resized)
        cv.waitKey(1)

        rawCapture.truncate(0)