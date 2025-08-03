import cv2 as cv
from utils import *
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
# Lectura de imágenes
imagen = cv.imread("yerba - 2020 (joya)/yerba.jpg")
imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
imagen_rotada = cv.imread("yerba - 2020 (joya)/yerba_rotada.jpg")
angulo = detectar_rotacion(imagen_rotada)
imagen_sin_rotar = rotate(imagen_rotada, angulo-90)
imagen_sin_rotar_gris = cv.cvtColor(imagen_sin_rotar, cv.COLOR_BGR2GRAY)
#le aplicamos un umbralado, luego con un kernel pasamos por el código de barras para solo dejar las barras
# _,imagen_binaria = cv.threshold(imagen_gris,1,255,cv.THRESH_BINARY)
# ut.trackbar_canny(imagen_gris)
cv.imshow("im",imagen_sin_rotar_gris)
cv.waitKey(0)
imagen_binaria = cv.Canny(imagen_sin_rotar_gris, 33,255)
# kernel = np.array([
#     [0,1,0],
#     [0,1,0],
#     [0,1,0]
# ], dtype=np.uint8)
kernel = np.zeros((5,5),dtype=np.uint8)
kernel[:,2] = 1

# imagen_binaria_invertida = cv.bitwise_not(imagen_binaria)
imagen_erosionada = cv.erode(imagen_binaria, kernel, iterations=9)
kernel_dilate = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], dtype=np.uint8)

# Este mejor si solo la imagen es bien limpia
# def extraer_ROI_mascara(mascara):
#     # Dada una imagen binaria (máscara)
#     y_indices, x_indices = np.nonzero(mascara)
#     x_min, x_max = x_indices.min(), x_indices.max()
#     y_min, y_max = y_indices.min(), y_indices.max()
#     roi = mascara[y_min:y_max+1, x_min:x_max+1]
#     return roi

#Este anda mejor imagenes que tienen porquería 
def extraer_ROI_mascara(mascara):
    rows, cols = mascara.shape
    top_left = (0, 0)
    bottom_right = (cols, rows)

    contours, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(largest_contour)

    top_left_roi = (x, y)
    bottom_right_roi = (x + w, y + h)

    roi = mascara[top_left_roi[1]:bottom_right_roi[1], top_left_roi[0]:bottom_right_roi[0]]

    return roi



imagen_dilatada = cv.dilate(imagen_erosionada, kernel_dilate, iterations=30)
codigo_barras = cv.bitwise_and(imagen_sin_rotar, imagen_sin_rotar, mask=imagen_dilatada)

codigo_barras_gris = cv.cvtColor(codigo_barras, cv.COLOR_BGR2GRAY)

_, codigo_barras_binaria = cv.threshold(codigo_barras_gris, 127, 255, cv.THRESH_BINARY)

cv.imshow("codigo_binario", codigo_barras_binaria)
cv.waitKey(0)
roi = extraer_ROI_mascara(codigo_barras_binaria)
roi = cv.bitwise_not(roi)
cv.imshow("Im", roi)
cv.waitKey(0)

# roi = extraer_ROI_mascara(imagen_dilatada)



# bordes = cv.Canny(imagen_recuperada, 33, 133)

# contornos, _ = cv.findContours(bordes, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# #mantener solo los más grandes
# contornos = sorted(contornos, key=cv.contourArea, reverse=True)[:10]

# contorno_codigo_barras = None
# umbral_relacion_aspecto_superior = 2.0
# umbral_relacion_aspecto_inferior = 0.5
# max_area = 0
# for contorno in contornos:
#     epsilon = 0.02*cv.arcLength(contorno, True) #perímetro del contorno
#     approx = cv.approxPolyDP(contorno, epsilon, True) #aproxima el contorno a otro contorno con menos vértices
#                                                       #epsilon especifica la precisión de la aproximación 
#                                                       # True indica contorno cerrado
    
#     cv.drawContours(imagen, [contorno], -1, (0, 255, 0), 2)
#     if len(approx) == 4: #rectángulo
#         # Como pueden haber varios rectángulos se puede:

#         #1. filtrar por área
#         area = cv.contourArea(contorno)
#         # if area > max_area:
#         #     max_area = area
#         #     contorno_codigo_barras = approx
#         #     break

#         #2. filtrar por relación de aspecto
#         x, y, w, h = cv.boundingRect(approx)
#         relacion_aspecto = float(w)/h
#         if relacion_aspecto > umbral_relacion_aspecto_inferior and relacion_aspecto < umbral_relacion_aspecto_superior:
#             if area > max_area:
#                 max_area = area
#                 contorno_codigo_barras = approx
#                 break
#         #3. Aplicar umbral de intensidad



