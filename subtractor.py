from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
cap = cv2.VideoCapture('Videos/test1.h264')
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows = False)

while(1):
    ret, frame = cap.read()
    if frame is not None: 
        frame = imutils.resize(frame, width=800)
        fgmask = fgbg.apply(frame)
        circles = cv2.HoughCircles(fgmask,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
        print(circles)
        # for i in circles[0,:]:
        #     # draw the outer circle
        #     cv2.circle(fgmask,(i[0],i[1]),i[2],(0,255,0),2)
        #     # draw the center of the circle
        #     cv2.circle(fgmask,(i[0],i[1]),2,(0,0,255),3)      
        cv2.imshow('frame',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()