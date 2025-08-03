import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import cv2 as cv
from utils import ventana_trackbars, filtro_butterworth_pasa_bajos, filtro_butterworth_pasa_altos

############################################################################################
# 1. Genere la funcion de transferencia H que caracteriza a un filtro homomorfico.
############################################################################################
def filtro_homomorfico(imagen, gL, gH, D0, orden=5):# gL: ganancia de baja frecuencia
    imagen_float = np.float32(imagen)               # gH: ganancia de alta frecuencia
#                                                   # D0: frecuencia de corte
    imagen_log = np.log1p(imagen_float)             # orden: orden del filtro                              
    
    imagen_TDF = fft.fftshift(fft.fft2(imagen_log))    

    pasa_bajos = filtro_butterworth_pasa_bajos(imagen.shape, D0, orden)
    pasa_altos = filtro_butterworth_pasa_altos(imagen.shape, D0, orden)

    imagen_pasa_altos_TDF = imagen_TDF * pasa_altos
    imagen_pasa_bajos_TDF = imagen_TDF * pasa_bajos
    
    imagen_pasa_altos = fft.ifft2(fft.ifftshift(imagen_pasa_altos_TDF))
    imagen_pasa_bajos = fft.ifft2(fft.ifftshift(imagen_pasa_bajos_TDF))
    
    imagen_filtrada = gH * np.exp(imagen_pasa_altos.real) - gL * np.exp(imagen_pasa_bajos.real)
    
    # Clip o normalize?
    # np.uint8(np.clip(imagen_filtrada, 0, 255))
    # imagen_final = cv.normalize(imagen_filtrada,  
    
    return imagen_filtrada

################################################################################################################################
# 2. Aplique el proceso en las imagenes ‘casilla.tif’ y ‘reunion.tif’, con valores apropiados de gL, gH, D0 y orden 
    #(prueba y error en cada imagen...).
################################################################################################################################
ruta1 = "Imagenes_Ej/casilla.tif"
ruta2 = "Imagenes_Ej/reunion.tif"

casilla = cv.imread(ruta1, cv.IMREAD_GRAYSCALE)
reunion = cv.imread(ruta2, cv.IMREAD_GRAYSCALE)

variables_trackbar = ['gL', 'gH', 'D0', 'orden']

parametros_trackbar = [[100, 2000], [100, 2000], [1, 100], [1, 10]]

def transformacion(imagen, valores_trackbar):
    gL = np.clip(valores_trackbar[0],100,2000)/1000 #de 0.1 a 2
    gH = np.clip(valores_trackbar[1],100,2000)/1000
    D0 = np.clip(valores_trackbar[2],1,100)/100 # de 0.1 a 1
    orden = np.clip(valores_trackbar[3],1,10) # de 1 a 10

    imagen_salida = filtro_homomorfico(imagen, gL, gH, D0, orden)
    
    return imagen_salida

ventana_trackbars(casilla,variables_trackbar, parametros_trackbar, transformacion)

gL = 0.5
gH = 0.5
D0 = 0.8
orden = 5
casilla_filtrada = filtro_homomorfico(casilla, gL, gH, D0, orden)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(casilla, cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(casilla_filtrada, cmap='gray')
plt.title('Imagen Filtrada')

plt.show()
# 3. Verifique las bondades del metodo comparando el resultado anterior con la imagen que se obtiene al ecualizar la imagen original. Esta tecnica suele ser eficaz combinada con alguna tecnica de manipulacion de histogramas, por ejemplo ecualizacion. Ecualice el resultado del filtrado y visualıcelo junto a los demas.