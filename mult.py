import numpy as np
import cv2
import sys
import imutils
import time
from centroidtracker import Centroid, CentroidTracker
from sklearn.svm import SVC


def training():
    images = []

    for index in range(1, 464):
        img = cv2.cvtColor(cv2.imread('Data/Train/video17/frame{}.jpg'.format(index)), cv2.COLOR_BGR2GRAY)
        images.append(cv2.resize(img, (20, 20)))

    # for index in range(1, 89):
    #     img = cv2.cvtColor(cv2.imread(
    #         'Data/Train/video18/frame{}.jpg'.format(index)), cv2.COLOR_BGR2GRAY)
    #     images.append(cv2.resize(img, (20, 20)))

    X = np.concatenate(images, axis=0)
    y = [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 100 : 199
    y += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 200 : 299
    y += [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]

    # 300 : 463
    y += [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
          0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # y += [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,              1, 1, 1, 1, 1, 1, 1, 1, 1, 0,                0, 0, 0, 0, 0, 0, 0, 1, 0, 1,          1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
    #       1, 1, 1, 1, 0, 1, 1, 1, 0, 1,           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                0, 1, 1, 1, 1, 1, 1, 0]

    y = np.array(y)
    Y = y.reshape(-1)
    X = X.reshape(len(y), -1)
    classifier_linear = SVC(kernel='linear')
    classifier_linear.fit(X, Y)

    return classifier_linear


video_path = '../Videos/video10.h264'
cv2.ocl.setUseOpenCL(False)

# version = cv2.__version__.split('.')[0]
# print(version)

# read video file
cap = cv2.VideoCapture(video_path)

# check opencv version
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=5, nmixtures=10, backgroundRatio=0.)

fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=300)
# centroids = CentroidTracker()
centers = np.array([])

m = (800.0 - 549.0) / (541.0 - 387.0)
number_frame = 0
kernel = np.ones((5, 5), np.uint8)

image_number = 1
classifier_linear = training()
while (True):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()
    number_frame += 1

    if frame is None:
        break
    if ret == True:

        frame = imutils.resize(frame, width=800, height=600)
        fgmask = cv2.GaussianBlur(frame, (7, 7), 0)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel)

        fgmask = fgbg.apply(fgmask)

        intensity, frameRemoveShadow = cv2.threshold(
            fgmask, 230, 255, cv2.THRESH_BINARY)

        # find object by movement
        (im2, contours, hierarchy) = cv2.findContours(
            frameRemoveShadow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # looping for contours
        # contours = [contour for contour in contours if cv2.contourArea(contour) > 500 and cv2.contourArea(contour) < 4000]
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            # elif np.sum(fgmask == 255) > 24000:
            #     continue
            x, y, w, h = cv2.boundingRect(contour)
            x_t = (y - 542.0 + m * 800.0)/m

            perimeter = cv2.arcLength(contour,True)

            # if(x > x_t or y > 541):
            #     continue
            # get bounding box from countour
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            initial_y = int(y-radius) if int(y-radius) > 0 else 0
            final_y = int(y+radius)
            initial_x = int(x-radius) if int(x-radius) > 0 else 0
            final_x = int(x+radius)

            crop_img = frame[initial_y:final_y, initial_x:final_x]
            crop_img = cv2.resize(crop_img, (20, 20))
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            prediction = classifier_linear.predict(crop_img.reshape(1, -1))

            if(prediction.item() == 0):
                continue

            M = cv2.moments(contour)

            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            np.append(centers, center)

            cv2.circle(frame, center, 15, (0, 255, 0), 2)
            epsilon = 0.01* cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
            

            # cv2.imshow("cropped", crop_img)
            # cv2.imwrite("Data/Train/video1/frame%d.jpg" % image_number, approx)

            # if image_number == 11:
            #     time.sleep(4)
            image_number += 1

            # centroids.update(center[0], center[1])
            # time.sleep(0.5)
            # draw bounding circle
        # if len(contours) is 0:
        #     centroids.setAllUndetected()
        #     centroids.removeAllUndetected()

        # time.sleep(0.1)
        # time.sleep(0.1)
        cv2.imshow('rgb', frame)
        cv2.imshow('foreground and background', frameRemoveShadow)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
