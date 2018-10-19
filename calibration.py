import cv2
import numpy as np
import imutils

 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Videos/teste_mesa_longo.mp4')
#cv2.imread

count = 0
 
while True:
    _, frame = cap.read()
 
    cv2.circle(frame, (550, 450), 5, (0, 0, 255), -1) #1 superior esquerdo
    cv2.circle(frame, (1400, 450), 5, (0, 0, 255), -1) #2 superior direito
    cv2.circle(frame, (150, 1000), 5, (0, 0, 255), -1) #3 inferior esquerdo
    cv2.circle(frame, (1700, 1000), 5, (0, 0, 255), -1) #4 inferior direito
 
    pts1 = np.float32([[550, 450], [1400, 450], [150, 1000], [1700, 1000]])
#    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    pts2 = np.float32([[0, 0], [600, 0], [0, 700], [600, 700]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, matrix, (750,750))
 
    frame = imutils.resize(frame, width=400)

    cv2.imshow("Frame", frame)
    

    cv2.imshow("Perspective transformation", result)
    cv2.imwrite("frame%d.jpg" % count,result)
    count +=1

 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()