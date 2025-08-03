from utils import mse, ruido_impulsivo_unimodal, ruido_gaussiano
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from icecream import ic
from scipy.ndimage import generic_filter
#---------------------------------------------Implementación de los filtros-------------------------------------------------
## Ambos funcionan pero creo va mejor el descomentado
# def filtro_media_geometrica(imagen, size=3):
#     filas, columnas = imagen.shape
#     imagen_filtrada = np.zeros_like(imagen, dtype=np.float32)
#     m, n = size, size
    
#     for x in range(filas):
#         for y in range(columnas):
#             producto = 1.0
#             for i in range(-(m // 2), m // 2 + 1):
#                 for j in range(-(n // 2), n // 2 + 1):
#                     if x + i >= 0 and x + i < filas and y + j >= 0 and y + j < columnas:
#                         producto *= imagen[x + i, y + j]
#             imagen_filtrada[x, y] = producto ** (1 / (m * n))
    
#     return imagen_filtrada.astype(np.uint8)

def filtro_media_geometrica(image, size):
    def geometric_mean(values):
        # Ensure there are no zero values to prevent division by zero
        values = values[values != 0]
        if len(values) == 0:
            return 0
        return np.prod(values) ** (1.0 / len(values))

    filtered_image = generic_filter(image, geometric_mean, size=size)
    return filtered_image

## Ambos filtros funcionan no se cual es más ideal
def filtro_media_contra_armonica(image, size=3, Q=1):
    def contraharmonic_mean(values, q):
        numerator = np.sum(values ** (q + 1))
        denominator = np.sum(values ** q)
        if denominator == 0:
            return 0
        return numerator / denominator

    contraharmonic_filter = lambda values: contraharmonic_mean(values, Q)

    filtered_image = generic_filter(image, contraharmonic_filter, size=size)
    return filtered_image

# def filtro_media_contra_armonica(imagen, size=3, Q=1):
#     numerador = cv.pow(imagen, Q + 1)
#     denominador = cv.pow(imagen, Q)
#     return cv.divide(cv.boxFilter(numerador, -1, (size, size)), cv.boxFilter(denominador, -1, (size, size)))

#---------------------------------------------Agregado de ruido-------------------------------------------------
imagen = cv.imread("Imagenes_Ej/sangre.jpg",cv.IMREAD_GRAYSCALE)

media = 0
stdv = 10
ruido_gaussiano = ruido_gaussiano(imagen.shape, media, stdv)

prob = 0.05
ruido_impulsivo = ruido_impulsivo_unimodal(imagen.shape,prob)

imagen_ruido = imagen + ruido_gaussiano + ruido_impulsivo

size_filtro = 3

imagen_ruido_filtro_media_geometrica = filtro_media_geometrica(imagen_ruido, size_filtro)
imagen_ruido_filtro_contra_armonica = filtro_media_contra_armonica(imagen_ruido, size_filtro)

#---------------------------------------------Comparación cualitativa-------------------------------------------------
fig, ax = plt.subplots(2,2, figsize=(15,10))

ax[0, 0].imshow(imagen, cmap='gray')
ax[0, 0].set_title("Imagen original")
ax[0, 0].axis('off')

# Subplot 2: Imagen con ruido
ax[0, 1].imshow(imagen_ruido, cmap='gray')
ax[0, 1].set_title("Imagen con ruido")
ax[0, 1].axis('off')

# Subplot 3: Ruido filtrado con media geométrica
ax[1, 0].imshow(imagen_ruido_filtro_media_geometrica, cmap='gray')
ax[1, 0].set_title("Ruido filtrado media geométrica")
ax[1, 0].axis('off')

# Subplot 4: Ruido filtrado con media contra armónica
ax[1, 1].imshow(imagen_ruido_filtro_contra_armonica, cmap='gray')
ax[1, 1].set_title("Ruido filtrado contra armónica")
ax[1, 1].axis('off')

plt.show()

#---------------------------------------------Comparación cuantitativa-------------------------------------------------
mse_ruido = mse(imagen,imagen_ruido)
mse_media_geometrica = mse(imagen, imagen_ruido_filtro_media_geometrica)
mse_contra_armonica = mse(imagen, imagen_ruido_filtro_contra_armonica)

print(f"MSE ruido: {mse_ruido}")
print(f"MSE media geométrica: {mse_media_geometrica}")
print(f"MSE contra armónica: {mse_contra_armonica}")