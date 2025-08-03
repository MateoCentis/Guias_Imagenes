from keyboard import record
import numpy as np
from scipy.ndimage import median_filter, generic_filter
import cv2 as cv
from utils import ruido_gaussiano, ruido_impulsivo_unimodal, mse
import matplotlib.pyplot as plt
def filtro_mediana(imagen, size):
    filtered_imagen = median_filter(imagen, size=size)
    return filtered_imagen

def filtro_punto_medio(imagen, size):
    min_filtered = generic_filter(imagen, np.min, size=size)
    max_filtered = generic_filter(imagen, np.max, size=size)
    filtered_imagen = (min_filtered + max_filtered) / 2
    return filtered_imagen

def filtro_media_alfa_recortado(imagen, size, d):
    def alpha_trimmed_mean(values, d):
        sorted_values = np.sort(values)
        trimmed_values = sorted_values[d:-d]
        return np.mean(trimmed_values)

    filtered_image = generic_filter(imagen, alpha_trimmed_mean, size=size, extra_keywords={'d': d})
    return filtered_image

imagen = cv.imread('Imagenes_Ej/sangre.jpg', cv.IMREAD_GRAYSCALE)

media = 0
stdv = 10
ruido_gaussiano = ruido_gaussiano(imagen.shape, media, stdv)

prob = 0.05
ruido_impulsivo = ruido_impulsivo_unimodal(imagen.shape,prob)

imagen_ruido = imagen + ruido_gaussiano + ruido_impulsivo
imagen_filtro_mediana = filtro_mediana(imagen_ruido, size=(3, 3))
imagen_filtro_punto_medio = filtro_punto_medio(imagen_ruido, size=(3, 3))
imagen_filtro_alfa_recortado = filtro_media_alfa_recortado(imagen_ruido, size=(3, 3), d=2)
imagen_filtro_mediana_punto_medio = filtro_punto_medio(imagen_filtro_mediana, size=(3,3))

mse_ruido = mse(imagen,imagen_ruido)
mse_mediana = mse(imagen, imagen_filtro_mediana)
mse_punto_medio = mse(imagen, imagen_filtro_punto_medio)
mse_alfa_recortado = mse(imagen, imagen_filtro_alfa_recortado)
mse_mediana_punto_medio = mse(imagen, imagen_filtro_mediana_punto_medio)

print(f"MSE ruido: {mse_ruido}")
print(f"MSE mediana: {mse_mediana}")
print(f"MSE punto medio: {mse_punto_medio}")
print(f"MSE alfa recortado: {mse_alfa_recortado}")
print(f"MSE mediana punto medio: {mse_mediana_punto_medio}")

# Crear los subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(imagen, cmap='gray')
axs[0, 0].set_title('Imagen Original')

axs[0, 1].imshow(imagen_ruido, cmap='gray')
axs[0, 1].set_title('Imagen con Ruido')

axs[0, 2].imshow(imagen_filtro_mediana, cmap='gray')
axs[0, 2].set_title('Filtro Mediana')

axs[1, 0].imshow(imagen_filtro_punto_medio, cmap='gray')
axs[1, 0].set_title('Filtro Punto Medio')

axs[1, 1].imshow(imagen_filtro_alfa_recortado, cmap='gray')
axs[1, 1].set_title('Filtro Alfa Recortado')

axs[1, 2].imshow(imagen_filtro_mediana_punto_medio, cmap='gray')
axs[1, 2].set_title('Filtro Mediana y Punto Medio')

plt.tight_layout()
plt.show()