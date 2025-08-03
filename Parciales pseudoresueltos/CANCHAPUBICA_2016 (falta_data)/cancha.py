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

img = cv.imread('./6.png')

imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#PREPARO LA IMAGEN PARA HALLAR LA LINEA HORIZONTAL
filtro = funx.Sobel('F')
suavizado = cv.filter2D(imgGRAY, -1, filtro)

#cv.imshow('suavizado', suavizado)

lHorizontal = funx.Hough(img, suavizado, 1)


#PREPARO LA IMAGEN PARA HALLAR LA LINEA VERTICAL
H = img.shape[0]
W = img.shape[1]

filtro = funx.Sobel('C')
suavizado = cv.filter2D(imgGRAY[int(H/2):H, :], -1, filtro)
suavizado[suavizado<255] = 0
#cv.imshow('suavizado', suavizado)

lVertical = funx.Hough(img, suavizado, 1)

#============================================================================================
#============================================================================================
#POSICIONO LAS IMAGENES
#============================================================================================
#============================================================================================

cv.imshow('imagen', img)
#cv.imshow('segmentada', img)

cv.waitKey(0)
