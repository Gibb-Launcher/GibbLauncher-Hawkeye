import cv2
import numpy as np
import imutils
import time

 
#cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../../Documents/PI2/Dia23/video001.h264')
cap = cv2.VideoCapture('videos/video002.h264')

count = 0
 
while True:
    _, frame = cap.read()
 
    if frame is not None:
        cv2.circle(frame, (175, 282), 5, (0, 0, 255), -1) #1 superior esquerdo
        cv2.circle(frame, (432, 291), 5, (0, 0, 255), -1) #2 superior direito
        cv2.circle(frame, (10, 390), 5, (0, 0, 255), -1) #3 inferior esquerdo
        cv2.circle(frame, (615, 405), 5, (0, 0, 255), -1) #4 inferior direito
    
        pts1 = np.float32([[175,282], [432, 291], [10,390], [615, 405]])
        #pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
        #pts2 = np.float32([[0, 0], [0, 300], [800, 0], [800, 300]])
        #pts2 = np.float32([[0, 0], [600, 0], [0, 700], [600, 700]])
        #pts2 = np.float32([[175,282], [432, 291], [10,390], [615, 405]])
        pts2 = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])

        # Ambos dão o mesmo resultado
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        h, status = cv2.findHomography(pts1,pts2)
        """ 
        print(matrix)
        print('====================')
        print(h)
        print("\n")
        """

        # Duas Maneiras de representar o ponto do quique do video 2
       # ponto_quique = np.matrix( [ [197.0],[427.0],[1.0] ] ) 
        ponto_quique = [197.0, 427.0, 1.0 ]

        x1 = np.matrix(matrix)
        x2 = np.array(ponto_quique)

        # Maneiras de calcular a matrix. matrix_calculada (1x3) = H(3x3) * ponto_quique(1x3)
        #matrix_calculada =  np.mat(teste) * np.mat(ponto_quique)
        matrix_calculada = x1.dot(x2)
        print(matrix_calculada)

        # Serve para aplicar a homografia na imagem
        result = cv2.warpPerspective(frame, h, (800,600))

        frame = imutils.resize(frame, width=800, height=600)
        result = imutils.resize(result, width=800, height=600)
        #cv2.circle(result, (547, 276), 5, (0, 255, 255), -1)

        cv2.imshow("Frame", frame)
        time.sleep(0.1)

        cv2.imshow("Perspective transformation", result)
    
    else:
        break

    
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()