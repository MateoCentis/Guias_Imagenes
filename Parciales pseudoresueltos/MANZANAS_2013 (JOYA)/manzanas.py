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

img = cv.imread('./img2.jpg')

verde = cv.imread('./cverde.jpg')
rojo = cv.imread('./croja.jpg')

(cVerde, _) = funx.datosPixel(verde, 1, 1, 1)
(cRojo, _) = funx.datosPixel(rojo, 1, 1, 1)

#(_, mVerde) = funx.segmentarColor(img, cVerde, 1)
#(_, mRojo) = funx.segmentarColor(img, cRojo, 1)

salida = funx.segmentarColorRGB(img, verde, 2)

mRojo = funx.segmentarColorRGB(img, rojo, 2)

salida = cv.medianBlur(salida, 5)
mRojo = cv.medianBlur(mRojo, 5)
# Cross-shaped Kernel
eE = cv.getStructuringElement(cv.MORPH_CROSS,(7,7))
salida1 = cv.erode(salida, eE)
eE = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
mRojo1 = cv.erode(mRojo, eE)
eE = np.ones((13,13))
#eE = cv.getStructuringElement(cv.MORPH_CROSS,(11,11))

salida1 = cv.dilate(salida1, eE)
mRojo1 = cv.dilate(mRojo1, eE)
#plt.figure("rojo erosionado")
#plt.imshow(mRojo1, 'gray')
#plt.show()
#salida1 = cv.medianBlur(salida1, 11)

plt.figure("original")
plt.imshow(img)

#plt.figure('segmentada Roja')
#plt.imshow(mRojo, 'gray')

#plt.figure('segmentada Verde')
#plt.imshow(salida, 'gray')

#plt.figure('Erosion')
#plt.imshow(salida1, 'gray')

resta = salida1-salida
resta2 = mRojo1-mRojo

resta = cv.dilate(resta, eE)

plt.figure('Resta Verde - Manzanas consideradas chicas')
plt.imshow(resta, 'gray')

plt.figure('Resta Rojo - Manzanas consideradas chicas')
plt.imshow(resta2, 'gray')

#DESPUES LO QUE RESTARIA SERIA VER QUE ELEMENTO ESTRUCTURANTE USAR PARA ASEGURAR QUE LAS MANZANAS
#ENTRE EN LA MASCARA, SACARLAS CON LA MASCARA Y LUEGO CAMBIARLES EL COLOR
#SE PODRIA CONSIDERAR UNA SUMA DE LAS MASCARAS Y LUEGO REEMPLAZAR TODOS LOS ROJOS Y VERDES POR
#AZUL... Y PARA CONTAR HAY UN ALGORITMO EN FUNX QUE CONTABA OBJETOS, PASARLE LA DILATACION
#DONDE APARECEN TODOS LOS ELEMENTOS, QUE SERIA MROJO Y MVERDE CREO
#Y A GRANDES RASGOS DEBERIA DE FUNCIONAR TODO

plt.show()
