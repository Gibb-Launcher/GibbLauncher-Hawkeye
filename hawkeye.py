import numpy as np
import cv2
import sys
import imutils
import time
from centroidtracker import Centroid, CentroidTracker
from sklearn.svm import SVC


def check_movement(shadow_frame, frame, balls_traking):
    (im2, contours, hierarchy) = cv2.findContours(
        shadow_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = np.empty((0, 3))
    for contour in contours:
        if cv2.contourArea(contour) < 20 and cv2.contourArea(contour) < 50:
            continue

        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        candidates = np.append(candidates, np.array(
            [np.array([int(center[0]), int(center[1]), int(timestamp)])]), axis=0)

    points = np.empty((0, 3))
    if candidates.shape[0] > 1:
        q1 = candidates[(candidates[:, 0] >= 0) & (candidates[:, 0] <= 200)]
        q2 = candidates[(candidates[:, 0] > 200) & (candidates[:, 0] <= 400)]
        q3 = candidates[(candidates[:, 0] > 400) & (candidates[:, 0] <= 600)]
        q4 = candidates[(candidates[:, 0] > 600) & (candidates[:, 0] <= 800)]

        if q1.shape[0] <= 2:
            points = np.append(points, q1, axis=0)
        if q2.shape[0] <= 2:
            points = np.append(points, q2, axis=0)
        if q3.shape[0] <= 2:
            points = np.append(points, q3, axis=0)
        if q4.shape[0] <= 2:
            points = np.append(points, q4, axis=0)
    elif candidates.shape[0] == 1:
        points = np.append(points, np.array(
            [np.array([candidates[0][0], candidates[0][1], candidates[0][2]])]), axis=0)

    for point in points:
        cv2.circle(frame,(int(point[0]), int(point[1])),4,255,5)
        balls_traking.append([int(point[0]), int(point[1]), int(point[2])])

    # print(points)


def mask_to_objects(frame, mask, array_test, timestamp, threshold=0):
    """
    applies a blob detection algorithm to the image
    Args:
        mask: image mask scaled between 0 and 255 
        threshold: min pixel intensity of interest
    Returns:
        list of objects [(x,y)]
    """

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = threshold
    params.maxThreshold = 255

    # params.filterByArea = True
    # params.minArea = 0
    # params.maxArea = 5000

    # params.filterByCircularity = False
    # params.filterByInertia = False
    # params.filterByConvexity = False
    # params.filterByColor = False
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(mask)

    objects = np.empty((0, 3))

    for k in keypoints:
        (x, y) = k.pt
        x = int(round(x))
        y = int(round(y))
        cv2.circle(frame, (x, y), 4, 255, 5)
        point = np.array([np.array([x, y, timestamp])])
        objects = np.append(point, point, axis=0)
        array_test.append([x, y, timestamp])

    points = np.empty((0, 3))
    if objects.shape[0] > 1:
        q1 = objects[(objects[:,0] >= 0) & (objects[:,0] <= 200)]
        q2 = objects[(objects[:,0] > 200) & (objects[:,0] <= 400)]
        q3 = objects[(objects[:,0] > 400) & (objects[:,0] <= 600)]
        q4 = objects[(objects[:,0] >  600) & (objects[:,0] <= 500)]

        if q1.shape[0] <= 3:
            points = np.append(points, q1, axis=0)
        if q2.shape[0] <= 3:
            points = np.append(points, q2, axis=0)
        if q3.shape[0] <= 3:
            points = np.append(points, q3, axis=0)
        if q4.shape[0] <= 3:
            points = np.append(points, q4, axis=0)

    for point in points:
        cv2.circle(frame,(int(point[0]), int(point[1])),4,255,5)

    return objects


def training():
    images = []

    for index in range(1, 464):
        img = cv2.cvtColor(cv2.imread(
            'Data/Train/video17/frame{}.jpg'.format(index)), cv2.COLOR_BGR2GRAY)
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


video_path = '../Dia23/video001.h264'
cv2.ocl.setUseOpenCL(False)

# version = cv2.__version__.split('.')[0]
# print(version)

# read video file
cap = cv2.VideoCapture(video_path)

# check opencv version
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=5, nmixtures=10, backgroundRatio=0.)

fgbg = cv2.createBackgroundSubtractorKNN(history=1)
# centroids = CentroidTracker()
centers = np.array([])

m = (800.0 - 549.0) / (541.0 - 387.0)
number_frame = 0
kernel = np.ones((5, 5), np.uint8)

crop_number = 1
classifier_linear = training()
timestamp = 0
array_test = []
while (True):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()
    number_frame += 1

    if frame is None:
        break
    if ret == True:

        frame = imutils.resize(frame, width=800, height=600)
        fgmask = cv2.GaussianBlur(frame, (5, 5), 0)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = cv2.morphologyEx(
            fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
        fgmask = cv2.dilate(fgmask, kernel, iterations=3)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)

        fgmask = fgbg.apply(fgmask)

        intensity, frameRemoveShadow = cv2.threshold(
            fgmask, 0, 255, cv2.THRESH_BINARY)

        check_movement(frameRemoveShadow, frame, array_test)
 
        cv2.imshow('foreground and background', frameRemoveShadow)
        cv2.imshow('rgb', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # if timestamp == 145:
        #     time.sleep(5)
        timestamp += 1

print(timestamp)
print(array_test, len(array_test))

cap.release()
cv2.destroyAllWindows()
