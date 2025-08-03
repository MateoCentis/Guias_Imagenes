import numpy as np
from scipy.fft import fft2, fftshift,ifft2, ifftshift
import matplotlib.pyplot as plt
import cv2 as cv
from utils import ventana_trackbars, trackbar_transformacion


#---------------------------------------------Funciones y lectura-------------------------------------------------
def filtro_pasa_altos(imagen):
    
    mascara_cruz = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]], dtype=np.float32)
    
    imagen_filtrada = cv.filter2D(imagen, -1, mascara_cruz)

    return imagen_filtrada

def filtro_alta_potencia(imagen_pasa_altos, A):

    H_alta_potencia = (A - 1) + imagen_pasa_altos

    return H_alta_potencia

def filtro_enfasis_alta_frecuencia(imagen_pasa_altos, a, b):
                                 
    
    H_enfasis_alta_frecuencia = a + b * imagen_pasa_altos

    return H_enfasis_alta_frecuencia

imagen = cv.imread('Imagenes_Ej/camaleon.tif',cv.IMREAD_GRAYSCALE)
imagen_TDF = fftshift(fft2(imagen))

#---------------------------------------------Filtrados HAP y HEAF-------------------------------------------------
imagen_pasa_altos = filtro_pasa_altos(imagen)

A = 1  # HAP
a = 1  # HEAF
b = 1    # HEAF

HAP = filtro_alta_potencia(imagen_pasa_altos,A)
HEAF = filtro_enfasis_alta_frecuencia(imagen_pasa_altos, a, b)

HAP_TDF = np.abs(fftshift(fft2(HAP)))

HEAF_TDF = np.abs(fftshift(fft2(HEAF)))

#------------------------------------------Visualización de resultados-------------------------------------------------
mostrar_subplots = False
if mostrar_subplots:
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].imshow(imagen, cmap='gray')
    axs[0, 0].set_title('Imagen Original')

    axs[0, 1].imshow(np.log(1 + np.abs(imagen_TDF)), cmap='gray')
    axs[0, 1].set_title('Transformada de Fourier de la Imagen Original')

    axs[1, 0].imshow(HAP, cmap='gray')
    axs[1, 0].set_title('Filtro de Alta Potencia')

    axs[1, 1].imshow(np.log(1 + HAP_TDF), cmap='gray')
    axs[1, 1].set_title('Transformada de Fourier del Filtro de Alta Potencia')

    axs[2, 0].imshow(HEAF, cmap='gray')
    axs[2, 0].set_title('Filtro de Énfasis en Alta Frecuencia')

    axs[2, 1].imshow(np.log(1 + HEAF_TDF), cmap='gray')
    axs[2, 1].set_title('Transformada de Fourier del Filtro de Énfasis en Alta Frecuencia')

    plt.tight_layout()
    plt.show()

#---------------------------------------------Uso de trackbars-------------------------------------------------


variables_trackbars = ['A']
parametros_trackbars = [[0, 15000]]

def transformacion(imagen, valores_trackbar):
    A = valores_trackbar[0]/100 
    imagen_pasa_altos = filtro_pasa_altos(imagen)
    H_alta_potencia = (A - 1) + imagen_pasa_altos

    return H_alta_potencia

ventana_trackbars(imagen, variables_trackbars, parametros_trackbars, transformacion)

variables_trackbars2 = ['a','b']
parametros_trackbars2 = [[0, 10000],[0, 10000]]

def transformacion2(imagen, valores_trackbar):
    a = valores_trackbar[0]/1000
    b = valores_trackbar[1]/1000
    imagen_pasa_altos = filtro_pasa_altos(imagen)
    H_alta_frecuencia = a + b*imagen_pasa_altos

    return H_alta_frecuencia

ventana_trackbars(imagen, variables_trackbars2, parametros_trackbars2, transformacion2)