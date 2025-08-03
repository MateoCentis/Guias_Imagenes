import cv2 as cv
import numpy as np

imagen = cv.imread("Morfologia_Im/lluviaEstrellas.jpg")
imagen_gris = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)

_,imagen_bin = cv.threshold(imagen_gris, 127,255,cv.THRESH_BINARY)

cv.imshow("d", imagen_bin)
cv.waitKey(0)

kernel = np.zeros((3,3), np.uint8)

kernel[0,2] = 1
kernel[1,1] = 1
kernel[2,0] = 1

imagen_erosion = cv.erode(imagen_bin,kernel,iterations = 4)

resultado = cv.bitwise_and(imagen,imagen, mask = imagen_erosion)
cv.imshow("e", resultado)

cv.waitKey(0)

