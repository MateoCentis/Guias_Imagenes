"""
En este archivo se busca utilizar cada recurso aprendido para mejorar la calidad de una imagen,
viéndolo por unidad.
"""
import cv2 as cv
import numpy as np
from utils import ventana_trackbars


#---------------------------------------------Lectura de la imagen-------------------------------------------------
texto = cv.imread("imagenes_varias/texto.jpg",cv.IMREAD_GRAYSCALE)
esqueleto = cv.imread("Imagenes_Ej/esqueleto.tif",cv.IMREAD_GRAYSCALE)

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
    # mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    # imagen_promedio = cv.filter2D(imagen, -1, mascara_promedio)
    imagen_difusa = cv.medianBlur(imagen,sizeMascara)

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
    # Alta potencia espacial 
    # imagen_salida = (A*imagen_eq - imagen_difusa).astype(np.uint8)
    # Alta potencia frecuencial

    return imagen_salida

#UNIDAD 1:Corrección S-R para mejorar contraste y brillo y ecualización de histograma
variables_trackbar = ["sizeMascara", "A"]
parametros_trackbar = [[2,10],[0,4000]]
# Unidad 2 : Filtrado espacial 
#Pasa_bajos (promediadores)
#Pasa_altos(mascaras suma = 0 y suma = 1)



# ventana_trackbars(imagen2, variables_trackbar, parametros_trackbar, transformacion2)
ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion2)

