import cv2 as cv
import numpy as np
from utils import ventana_trackbars, evitar_desborde

ruta = "Imagenes_Ej/esqueleto.tif"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
cv.imshow("imagen",imagen)
cv.waitKey(0)
#Vamos a aplicar un filtro de acentuado

def transformacion(imagen2, valores_trackbar):
    imagen = imagen2.copy()
    sizeMascara = valores_trackbar[0]
    A = (valores_trackbar[1]+1000)/1000 #2: max, 1: min
    brillo = int((valores_trackbar[2]+100)/100)
    if sizeMascara < 1:
        sizeMascara = 1
    if A < 1:
        A = 1
    if brillo < 1:
        brillo = 0
    if brillo > 255:
        brillo = 255
    imagen += brillo
    #La imagen tiene mucho ruido: 
        #filtro de promediado + alta potencia (saco el ruido y devuelvo bordes con alta potencia)
    #Filtro de promediado
    # mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    # imagen_promedio = cv.filter2D(imagen, -1, mascara_promedio)
    imagen_difusa = cv.GaussianBlur(imagen,(sizeMascara,sizeMascara),0)

    imagen_salida = A*imagen_difusa - imagen_difusa
    return imagen_salida


variables_trackbar = ["sizeMascara", "A","Brillo"]
parametros_trackbar = [[1,10],[0,1000],[0,2550]]

ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)

