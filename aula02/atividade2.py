# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np 
from ipywidgets import widgets, interact, interactive, FloatSlider, IntSlider
import time

#Calculei o f usando a distância entre os centros dos círculos. 
#A distância do papel e da câmera e os pixels no paint da d centros.
#formula: f = d papel camera / d centro c * pixels
f = 545
d = 14

#Fonte
font = cv2.FONT_HERSHEY_SIMPLEX

#Cores hsv dos círculos a serem detectados
magentahsv1 = np.array([166,  100,  100], dtype= np.uint8)
magentahsv2 = np.array([176, 255, 255], dtype= np.uint8)
azulhsv1 = np.array([80, 100, 100], dtype= np.uint8)
azulhsv2 = np.array([147, 255, 255], dtype= np.uint8)

#Parâmetros usados ao abrir a webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1


# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)

    # Passar o espectro de cores para hsv
    maskazul = cv2.inRange(hsv, azulhsv1, azulhsv2)
    maskmagenta = cv2.inRange(hsv, magentahsv1, magentahsv2)
    mask = cv2.bitwise_or(maskazul, maskmagenta)
    
    circles = []

    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.bitwise_and(frame, frame, mask)

    # HoughCircles - detects circles using the Hough Method.
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=15,maxRadius=60)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        circles = circles[0]
        for i in circles:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)

            if len(circles) > 1:
                circle1 = circles[0]
                circle2 = circles[1]
            else:
                circle1 = circles[0]
                circle2 = circle1
            
        cv2.line(bordas_color,(circle1[0], circle1[1]), (circle2[0], circle2[1]), (0,0,255), 3)

        if len(circles) >= 2:
            deltax = abs(circle2[0] - circle1[0])
            deltay = abs(circle2[1] - circle1[1])
            dist = math.sqrt((deltax*deltax) + (deltay*deltay))
            dezao = (d*f)/(abs(dist))
            angulo = abs(math.atan2(deltay, deltax))
            angulop = math.degrees(angulo)
            cv2.putText(bordas_color, 'angulo = {}'.format(angulop), (0,110), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(bordas_color, 'distancia = {}'.format(dezao), (0,70), font, 1,(255,255,255),2,cv2.LINE_AA)
    

    cv2.putText(bordas_color,'Press q to quit',(0,30), font, 1,(255,255,255),2,cv2.LINE_AA)
 

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()