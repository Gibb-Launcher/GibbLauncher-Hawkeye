import numpy as np
import cv2
import sys
import imutils
import time
from centroidtracker import Centroid, CentroidTracker

video_path = 'Videos/test1.h264'
cv2.ocl.setUseOpenCL(False)

# version = cv2.__version__.split('.')[0]
# print(version)

# read video file
cap = cv2.VideoCapture(video_path)

# check opencv version
fgbg = cv2.createBackgroundSubtractorKNN()
fgbg.setShadowValue(230)
fgbg.setDetectShadows(True)

centroids = CentroidTracker()

while (True):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if ret == True:

        frame = imutils.resize(frame, width=800)

        fgmask = fgbg.apply(frame)

        intensity, frameRemoveShadow = cv2.threshold(fgmask, 230, 255, cv2.THRESH_BINARY)

        # find object by movement
        (im2, contours, hierarchy) = cv2.findContours(
            frameRemoveShadow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # looping for contours
        contours = [contour for contour in contours if cv2.contourArea(contour) > 500 and cv2.contourArea(contour) < 4000]
        for contour in contours:
            if cv2.contourArea(contour) < 500 or cv2.contourArea(contour) > 4000 :
                continue

            # get bounding box from countour
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            M = cv2.moments(contour)

            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            initial_y = int(y-radius) if int(y-radius) > 0 else 0
            final_y = int(y+radius)
            initial_x = int(x-radius) if int(x-radius) > 0 else 0
            final_x = int(x+radius)
 
            crop_img = frame[initial_y:final_y, initial_x:final_x]
            cv2.imshow("cropped", crop_img)
            centroids.update(center[0], center[1])
            # time.sleep(0.5)
            # draw bounding circle
            cv2.circle(frame, center, 15, (0, 255, 0), 2)
        if len(contours) is 0:
            centroids.setAllUndetected()
            centroids.removeAllUndetected()


        # time.sleep(0.1)
        cv2.imshow('foreground and background', frameRemoveShadow)
        cv2.imshow('rgb', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
