import numpy as np
import cv2 as cv
from utils import calcular_histogramas, graficar_histogramas_subplots, mostrar_imagenes, ventana_trackbars, trackbar_transformacion
import matplotlib.pyplot as plt


imagenA = cv.imread("Imagenes_Ej/FAMILIA_a.jpg", cv.IMREAD_GRAYSCALE)
imagenB = cv.imread("Imagenes_Ej/FAMILIA_b.jpg", cv.IMREAD_GRAYSCALE)
imagenC = cv.imread("Imagenes_Ej/FAMILIA_c.jpg", cv.IMREAD_GRAYSCALE)

ver_imagenes = False
if ver_imagenes:
    mostrar_imagenes([imagenA, imagenB, imagenC])

    histogramas = calcular_histogramas([imagenA, imagenB, imagenC])
    graficar_histogramas_subplots(histogramas)
 
# Identificación de tipo de ruido
#imagen A -> ruido gaussiano
#imagen B -> ruido uniforme
#imagen C -> ruido sal y pimienta

# Elección de filtro
# fastNlMeansDenoising() #Funciona bien contra ruido impulsivo y Gaussiano 
    #h: (más alto más ruido saca pero más imagen se pierde, 10 ta ok)
    #hForColorComponents: lo mismo pero para cada color
    #templateWindowSize: debe ser impar (7 recomendado)
    #searchWindowSize: debe ser impar (21 recomendado)

# h = 20
# templateWindowSize = 7
# searchWindowSize = 21
# imagen_filtrada = cv.fastNlMeansDenoising(imagenA,None,h,templateWindowSize,searchWindowSize)
imagen_a = True
if imagen_a:
    variables_trackbar = ['h', 'templateWindowSize', 'searchWindowSize']

    parametros_trackbar = [[1,100],[3,21],[3,21]]

    def transformacion(imagen, valores_trackbar):
        h = valores_trackbar[0]
        templateWindowSize = valores_trackbar[1]
        searchWindowSize = valores_trackbar[2]
        imagen_filtrada = cv.fastNlMeansDenoising(imagen,None,h,templateWindowSize,searchWindowSize)
        return imagen_filtrada

    trackbar_transformacion(imagenA,variables_trackbar, parametros_trackbar, transformacion)

imagen_b = True
if imagen_b:
    variables_trackbar = ['h', 'templateWindowSize', 'searchWindowSize']

    parametros_trackbar = [[1,100],[3,21],[3,21]]

    def transformacion(imagen, valores_trackbar):
        h = valores_trackbar[0]
        templateWindowSize = valores_trackbar[1]
        searchWindowSize = valores_trackbar[2]
        imagen_filtrada = cv.fastNlMeansDenoising(imagen,None,h,templateWindowSize,searchWindowSize)
        return imagen_filtrada

    trackbar_transformacion(imagenB,variables_trackbar, parametros_trackbar, transformacion)

imagen_c = True
if imagen_c:
    # Como la imagen c tiene ruido sal y pimienta debemos buscar otro filtro
    # ksize = 3 #best
    # imagen_filtrada = cv.medianBlur(imagenC, ksize)
    # mostrar_imagenes([imagenC, imagen_filtrada])

    variables_trackbar = ['k_size']
    parametros_trackbar = [[1,30]]
    def transformacion(imagen, valores_trackbar):
        k_size = valores_trackbar[0]
        imagen_filtrada = cv.medianBlur(imagen, k_size)
        return imagen_filtrada

    trackbar_transformacion(imagenC,variables_trackbar, parametros_trackbar, transformacion)