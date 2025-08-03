import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import ventana_trackbars, evitar_desborde
def transformacion2(imagen2, valores_trackbar):
    imagen = imagen2.copy()
    sizeMascara = valores_trackbar[0]
    A = (valores_trackbar[1]+1000)/1000 #2: max, 1: min
    if sizeMascara < 2:
        sizeMascara = 2
    if sizeMascara % 2 == 0:
        sizeMascara += 1
    if A < 1:
        A = 1
    #La imagen tiene mucho ruido: 
        #filtro de promediado + alta potencia (saco el ruido y devuelvo bordes con alta potencia)
    #Filtro de promediado
    mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    imagen_difusa = cv.filter2D(imagen, -1, mascara_promedio)
    # imagen_difusa = cv.medianBlur(imagen,sizeMascara)
    # Alta potencia
    imagen_salida = (A*imagen - imagen_difusa).astype(np.uint8)
    # Ecualizado
    imagen_salida = cv.equalizeHist(imagen_salida)
    return imagen_salida

def transformacion3(imagen2, valores_trackbar):
    imagen = imagen2.copy()
    sizeMascara = valores_trackbar[0]
    A = (valores_trackbar[1]+1000)/1000 #2: max, 1: min
    if sizeMascara < 2:
        sizeMascara = 2
    if sizeMascara % 2 == 0:
        sizeMascara += 1
    if A < 1:
        A = 1

    # Ecualizado
    imagen_eq = cv.equalizeHist(imagen)

    imagen_difusa = cv.medianBlur(imagen_eq,sizeMascara)
    # Alta potencia
    imagen_salida = (A*imagen_eq - imagen_difusa).astype(np.uint8)
    return imagen_salida

# ruta = "Imagenes_Ej/rosas.jpg"
ruta = "imagenes_varias/texto.jpg"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

ruta2 = "Imagenes_Ej/esqueleto.tif"
imagen2 = cv.imread(ruta2,cv.IMREAD_GRAYSCALE)
# cv.imshow("imagen",imagen)
# cv.waitKey(0)
#Vamos a aplicar un filtro de acentuado
ventana = False
if ventana:


    variables_trackbar = ["sizeMascara", "A","Brillo"]
    parametros_trackbar = [[2,10],[0,1000],[0,2550]]

    ventana_trackbars(imagen2, variables_trackbar, parametros_trackbar, transformacion2)


def transformacion(imagen2, valores_trackbar):
    imagen = imagen2.copy()
    sizeMascara = valores_trackbar[0]
    A = valores_trackbar[1] #2: max, 1: min
    if sizeMascara < 2:
        sizeMascara = 2
    if sizeMascara % 2 == 0:
        sizeMascara += 1
    if A < 1:
        A = 1
    #La imagen tiene mucho ruido: 
        #filtro de promediado + alta potencia (saco el ruido y devuelvo bordes con alta potencia)
    #Filtro de promediado
    # mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    # imagen_promedio = cv.filter2D(imagen, -1, mascara_promedio)
    imagen_difusa = cv.medianBlur(imagen,sizeMascara)

    # imagen_salida = cv.cvtColor(A*imagen_difusa - imagen_difusa, cv.COLOR_BGR2GRAY)
    # imagen_salida = cv.normalize(A*imagen - imagen_difusa,None,0,255,cv.NORM_MINMAX).astype(np.uint8)
    imagen_salida = (A*imagen - imagen_difusa).astype(np.uint8)
    return imagen_salida


sizeMascara = 3
A = 1
imagen_AP = transformacion(imagen, [sizeMascara,A,0])
imagen_eq = cv.equalizeHist(imagen)

imagen_AP_eq = cv.normalize(cv.equalizeHist(imagen_AP),None,0,255,cv.NORM_MINMAX).astype(np.uint8)
imagen_eq_AP = cv.normalize(transformacion(imagen_eq, [sizeMascara,A,0]),None,0,255,cv.NORM_MINMAX).astype(np.uint8)

cv.imshow("Alta potencia, luego ecualizada", imagen_AP_eq)
cv.imshow("Ecualizada, luego alta potencia", imagen_eq_AP)

cv.waitKey(0)

variables_trackbar = ["sizeMascara", "A"]
parametros_trackbar = [[2,10],[0,4000]]

# ventana_trackbars(imagen2, variables_trackbar, parametros_trackbar, transformacion2)
ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion2)

