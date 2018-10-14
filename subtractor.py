from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
cap = cv2.VideoCapture('Videos/test1.h264')
fgbg = cv2.createBackgroundSubtractorMOG2()

#Lower and upper boundaries
greenLower = (0, 0, 255)
greenUpper = (255, 15, 255)

# List of all points (x,y)
pts = deque(maxlen=5000)

bateu = 0
mudou = False

while(1):
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=600)
    
    fgmask = fgbg.apply(frame)
#    hsv = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)


    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    #mask = cv2.inRange(frame, greenLower, greenUpper)
    #mask = cv2.erode(mask, None, iterations=2)
    #mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()