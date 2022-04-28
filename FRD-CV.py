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
##print("MARKER_LENGTH_IN = ", MARKER_LENGTH_IN)

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
##    camera.awb_mode = 'auto'
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
##        frame.awb_mode = 'auto'
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
        i = 0
        j = 0
        keepIDs = []
        keepCorners = []

        # Wait to hit target minimum (Stage 4)
        if setMax == True:
            if setMin == False:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        if tag == 2:
                            count = count + 1
##                            print("Aruco 2 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner2 in corners:
                                if j == i:
                                    keepCorners.append(corner2)
                                j = j + 1
                        if tag == 3:
                            count = count + 1
##                            print("Aruco 3 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner3 in corners:
                                if j == i:
                                    keepCorners.append(corner3)
                                j = j + 1
                        i = i + 1

                if count >= 2:
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
                    strMin = str(minDist)
                    cv.putText(img,
                               "Control Minimum: " + strMin,
                               (5, 150),
                               cv.FONT_HERSHEY_SIMPLEX,
                               2,
                               (0, 255, 0),
                               2
                               )
                    
                    if currDist >= minDist:
                        setMin = True
                        cv.putText(minSetImg,
                                   "Actuated Min Distance: " + strDist,
                                   (5, 60),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 0, 255),
                                   2
                                   )
                        cv.putText(minSetImg,
                                   "Control Min Distance: " + strMin,
                                   (5, 30),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 255, 0),
                                   2
                                   )
                        print("Actuated Min Distance = ", currDist)
                        cv.imshow("Actuated Min Img", minSetImg)
                        cv.moveWindow("Actuated Min Img", 1080, 720)
                        cv.waitKey(0)
                        print("CALIBRATION COMPLETE!")
                        

        # Wait to hit target maximum (Stage 3)
        if gotMin:
            if not setMax:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        if tag == 2:
                            count = count + 1
##                            print("Aruco 2 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner2 in corners:
                                if j == i:
                                    keepCorners.append(corner2)
                                j = j + 1
                        if tag == 3:
                            count = count + 1
##                            print("Aruco 3 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner3 in corners:
                                if j == i:
                                    keepCorners.append(corner3)
                                j = j + 1
                        i = i + 1

                if count >= 2:
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
                    strMax = str(maxDist)
                    cv.putText(img,
                               "Control Maximum: " + strMax,
                               (5, 50),
                               cv.FONT_HERSHEY_SIMPLEX,
                               1.5,
                               (0, 255, 0),
                               2
                               )
                    
                    if currDist <= maxDist:
                        setMax = True
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
                        cv.putText(maxSetImg,
                                   "Control Max Distance: " + strMax,
                                   (5, 30),
                                   cv.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (0, 255, 0),
                                   2
                                   )
                        cv.imshow("Actuated Max Img", maxSetImg)
                        cv.moveWindow("Actuated Max Img", 1080, 52)
                        cv.waitKey(0)

                        print("STAGE 4: SET MINIMUM")

        # Get maximum control distance (Stage 2)
        if gotMax:
            if not gotMin:
                if ids is not None:
                    keepCorners = []
                    for tag in ids:
                        j = 0
                        if tag == 0:
                            count = count + 1
##                            print("Aruco 0 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner0 in corners:
                                if j == i:
                                    keepCorners.append(corner0)
                                j = j + 1
                        if tag == 1:                        
                            count = count + 1
##                            print("Aruco 1 Detected")
                            keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                            for corner1 in corners:
                                if j == i:
                                    keepCorners.append(corner1)
                                j = j + 1
                        i = i + 1

                # When two markers are detected, get distance and show image                        
                if count >= 2:
                    minDist, minImg = get_dist(img,
                                               keepIDs,
                                               keepCorners,
                                               newCamMtx
                                               )
                    print("Minimum Distance: ", minDist)
                    gotMin = True

                    cv.imshow("Min Control Img", minImg)
                    cv.moveWindow("Min Control Img", 0, 720)
                    cv.waitKey(0)

                    # Print calibration distance range and difference
                    print("Control Distance Range: ",
                          round(minDist, 4), " - ",
                          round(maxDist, 4), " inches")
                    print("Control Distance Difference : ",
                          round(maxDist - minDist, 4), " inches")

                    print("STAGE 3: SET MAXIMUM")


        # Get maximum control distance (Stage 1)
        if not gotMax:
            if ids is not None:
                keepCorners = []
                for tag in ids:
##                    print("i: ", i)
                    j = 0
                    if tag == 0:
                        count = count + 1
##                        print("Aruco 0 Detected")
                        keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                        for corner0 in corners:
##                            print("j: ", j)
                            if j == i:
                                keepCorners.append(corner0)
##                                print("ARUCO 0 CORNER APPEND: ", keepCorners)
                            j = j + 1
                    if tag == 1:
                        count = count + 1
##                        print("Aruco 1 Detected")
                        keepIDs = np.append(keepIDs, [tag[0]], axis = 0)
                        for corner1 in corners:
##                            print("j: ", j)
                            if j == i:
                                keepCorners.append(corner1)
##                                print("ARUCO 1 CORNER APPEND: ", keepCorners)
                            j = j + 1
                    
##                    print("ids: ", ids)
##                    print("keepIDs: ", keepIDs)
##                    print("corners: ", corners)
##                    print("keepCorners: ", keepCorners)
                    i = i + 1
                    
##            print("----------")
##            print("----------")

            # When two markers are detected, get distance and show image        
            if count >= 2:
                maxDist, maxImg = get_dist(img,
                                           keepIDs,
                                           keepCorners,
                                           newCamMtx
                                           )
                print("Maximum Distance: ", maxDist)
                gotMax = True

                cv.imshow("Max Control Img", maxImg)
                cv.moveWindow("Max Control Img", 0, 52)
                cv.waitKey(0)

                print("STAGE 2: GET MINIMUM")


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
