import numpy as np
import cv2 as cv
import sys
import math
from time import time

sys.path.append('../../')

import funciones as funx
import pdifunFixed as funxPDI
import pdifunFixed as pdi
from matplotlib import pyplot as plt

#============================================================================================
#============================================================================================

img = cv.imread('./1.png')

H = img.shape[0]
W = img.shape[1]

(aux, maska) = funx.segmentarColor(cv.GaussianBlur(img, (7,7), 0), 180, 4)

r = 200-34

aux = funx.generarImagenVacia(H,W)
aux1 = funx.generarImagenVacia(H,W)

cv.circle(aux, (int(H/2), int(W/2)), r, 255, 10)
cv.circle(aux1, (int(H/2), int(W/2)), 118, 255, 20)

mask = funx.multiplicacion_mascara(maska, aux)
mask1 = funx.multiplicacion_mascara(maska, aux1)

#OBTENGO LA POSICION DEL NORTE
divisionesH = funx.divisionesH(mask, 5)
y = np.average(divisionesH)

divisionesV = funx.divisionesV(mask, 5)
x = np.average(divisionesV)

Norte = np.array([x, y])

#OBTENGO LA POSICION DE LA FLECHA
divisionesH = funx.divisionesH(mask1, 5)
y = np.average(divisionesH)

divisionesV = funx.divisionesV(mask1, 5)
x = np.average(divisionesV)

Brujula = np.array([x, y])
Centro = np.array([W/2, H/2])

print(Norte)
print(Centro)
print(Centro-Norte)

print("x: " + str(x))
print("y: " + str(y))

resultado = (Centro-Norte) * (Centro-Brujula) / ( funx.distancia(Centro, Norte) * funx.distancia(Centro, Brujula))
final = resultado[0] + resultado[1]
#print(final)

angulo = math.acos(final)

print("Angulo en radianes: " + str(angulo))
print("Angulo en grados: " + str(angulo*180/math.pi))

plt.figure('mascara')
plt.imshow(mask1, 'gray')

cv.imshow('original', img)
cv.imshow('segmentada', aux1)

cv.waitKey(0)
plt.show()
