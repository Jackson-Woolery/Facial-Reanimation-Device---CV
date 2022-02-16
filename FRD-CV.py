from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import cv2 as cv
import numpy as np
import math

WIDTH, HEIGHT = 1920, 1088
width, height = str(WIDTH), str(HEIGHT)

scale = 0.25
width_d = int(WIDTH * scale)
height_d = int(HEIGHT * scale)
dim = (width_d, height_d)

MARKER_LENGTH_IN = 4.5 / 25.4 # "5mm Marker"

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

KD = np.load('CV_CameraCalibrationData.npz')
K = KD['k']
DIST = KD['dist']

def get_dist(img, ids, corners, newCamMtx):
    
    # Get rotation and translation vectors with respect to marker
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        corners,
        markerLength=MARKER_LENGTH_IN,
        cameraMatrix=newCamMtx,
        distCoeffs=0
        )

    print("tvecs: ", tvecs)

    tvec0 = tvecs[0][0]
    tvec1 = tvecs[1][0]
    print("tvec0: ", tvec0)
    print("tvec1: ", tvec1)

    distance = math.sqrt((tvec0[0] - tvec1[0]) ** 2 +
                         (tvec0[1] - tvec1[1]) ** 2 + 
                         (tvec0[2] - tvec1[2]) ** 2
                         )

    cv.aruco.drawDetectedMarkers(image=img,
                                 corners=corners,
                                 ids=ids,
                                 borderColor=(0, 0, 255)
                                 )

    disp_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    return distance, disp_img


if __name__ == '__main__':
    # Initialize Camera
    camera = PiCamera()
    camera.resolution = (WIDTH, HEIGHT)
    camera.exposure_mode = 'sports'
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

    gotMin = False
    gotMax = False

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
        corr_img = cv.undistort(img,
                                K,
                                DIST,
                                None,
                                newCamMtx
                                )

        # Detect Aruco marker ids and corners
        corners, ids, _ = cv.aruco.detectMarkers(image=img,
                                                 dictionary=arucoDict,
                                                 cameraMatrix=K,
                                                 distCoeff=DIST
                                                 )
        
        count = 0

        # Get minimum control distance
        if gotMin == False:
            if ids is not None:
                for tag in ids:
                    count = count + 1
                    print(count, " tag detected (min)")
                    
            if count >= 2:
                minDist, minImg = get_dist(img, ids, corners, newCamMtx)
                print("Minimum Distance: ", minDist)
                gotMin = True

                cv.imshow("Min. Control Img", minImg)
                cv.waitKey(0)

        # Get maximum control distance
        if gotMin == True:
            if gotMax == False:
                if ids is not None:
                    for tag in ids:
                        count = count + 1
                        print(count, " tag detected (max)")
                        
                if count >= 2:
                    maxDist, maxImg = get_dist(img, ids, corners, newCamMtx)
                    print("Maximum Distance: ", maxDist)
                    gotMax = True

                    cv.imshow("Max. Control Img", maxImg)
                    cv.waitKey(0)
                    

        # Get live output values
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
