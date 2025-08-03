import cv2 as cv
import numpy as np
imagen = cv.imread("Morfologia_Im/estrellas.jpg")
imagen_gris = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)

_,imagen_bin = cv.threshold(imagen_gris, 120,255,cv.THRESH_BINARY)


cv.imshow("s",imagen_bin)
cv.waitKey(0)

#Hacer erosión para eliminar estrellas chiquitas
kernel = np.ones((3,3), np.uint8)

imagen_erosion = cv.erode(imagen_bin,kernel,iterations = 4)

resultado = cv.bitwise_and(imagen,imagen, mask = imagen_erosion)
cv.imshow("e", resultado)

cv.waitKey(0)