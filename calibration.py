import cv2
import numpy as np
 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Videos/mesa_ping.mp4')

 
while True:
    _, frame = cap.read()
 
    cv2.circle(frame, (155, 120), 5, (0, 0, 255), -1) #1 superior esquerdo
    cv2.circle(frame, (450, 120), 5, (0, 0, 255), -1) #2 superior direito
    cv2.circle(frame, (60, 300), 5, (0, 0, 255), -1) #3 inferior esquerdo
    cv2.circle(frame, (609, 332), 5, (0, 0, 255), -1) #4 inferior direito
 
    pts1 = np.float32([[155, 120], [480, 120], [20, 475], [620, 475]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
 
    result = cv2.warpPerspective(frame, matrix, (500,500))
 
 
    cv2.imshow("Frame", frame)
    
    
    cv2.imshow("Perspective transformation", result)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()