import numpy as np
import cv2 as cv
import sys
from time import time

sys.path.append('../../')

import funciones as funx
import pdifunFixed as funxPDI
import pdifunFixed as pdi
from matplotlib import pyplot as plt

#============================================================================================
#============================================================================================

img = funx.promediadoImagenes(1,22)
cv.imshow('Imagen Promediada', img)

H = img.shape[0]
W = img.shape[1]

filtro = funx.Laplaciano('N8')

gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

filtrada = cv.GaussianBlur(gris, (5,5), 0)
bordes = cv.Canny(filtrada, 80, 140)
cv.imshow('iman', bordes)

(_, theta) = funx.Hough(img, bordes, 0)

print("Angulo linea encontrada: " + str(theta) + " (en radianes)")
#de ese angulo se hacen calculos y mas o menos se rota

cv.waitKey(0)
