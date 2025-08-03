#2. Estudie la implementacion de la TH para cırculos cv.HoughCircles. 
#Utilizando la imagen ’latas.png’, realice un programa que: 
    #• cuente e informe el numero de latas
    #• que informe el numero de latas discriminando en grandes y pequenas
#Realice los preprocesamientos que crea necesarios, y puede probar la robustez de su implementacion rotando la imagen (180 grados).

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import trackbar_transformacion

latas = cv.imread('Imagenes_Ej/latas.png')
latas_rotadas = cv.rotate(latas, cv.ROTATE_180)

latas_gris = cv.cvtColor(latas, cv.COLOR_BGR2GRAY)
latas_gris_rotadas = cv.cvtColor(latas_rotadas, cv.COLOR_BGR2GRAY)
latas_blur = cv.GaussianBlur(latas_gris_rotadas, (9,9), 2)
# latas_blur_rotadas = cv.GaussianBlur(latas_gris_rotadas, (9,9), 2)

def transformacion(imagen, valores_trackbar):
    imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    imagen_salida = imagen_gris.copy()
    dp = valores_trackbar[0]/10
    minDist = valores_trackbar[1]
    param1 = valores_trackbar[2]
    param2 = valores_trackbar[3]
    minRadius = valores_trackbar[4]
    maxRadius = valores_trackbar[5]
    imagen_gris = cv.GaussianBlur(imagen_gris, (9,9), 4)
    circulos = cv.HoughCircles(imagen_gris, cv.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0,:]:
            # draw the outer circle
            cv.circle(imagen_salida,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(imagen_salida,(i[0],i[1]),2,(0,0,255),3)
 
    return imagen_salida

variables_trackbar = ['dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius']
parametros_trackbar = [[10,20],[1,1000], [1, 1000], [1, 1000], [1, 1000], [1, 1000]]
trackbar_transformacion(latas, variables_trackbar, parametros_trackbar, transformacion)

# circulos = cv.HoughCircles(latas_blur, cv.HOUGH_GRADIENT, dp=1, minDist=200,
#                            param1=100, param2=70, minRadius=10, maxRadius=500)
#     # Parámetros 
#         # image: La imagen en escala de grises.
#         # method: El método de detección. Normalmente se usa cv2.HOUGH_GRADIENT.
#         # dp: El inverso de la resolución de la acumulación. dp=1 significa usar la misma resolución que la imagen original.
#         # minDist: Distancia mínima entre los centros de los círculos detectados.
#         # param1: Parámetro de paso al método de detección de bordes (e.g., umbral de Canny).
#         # param2: Umbral de acumulación para el centro del círculo. Cuanto menor sea, más fácil será detectar círculos, pero también aumentará la posibilidad de falsos positivos.
#         # minRadius: Radio mínimo del círculo que se quiere detectar.
#         # maxRadius: Radio máximo del círculo que se quiere detectar.
# if circulos is not None:
#     circulos = np.round(circulos[0,:]).astype(np.uint8)

#     latas_salida = latas_rotadas.copy()

#     circulos_chicos = []
#     circulos_grandes = []

#     for (x, y, r) in circulos:
#         if r < 120:
#             circulos_chicos.append((x, y, r))
#         else:
#             circulos_grandes.append((x, y, r))
        
#         cv.circle(latas_salida, (x,y), r, (0,255,0), 4)
#         # cv.rectange(latas_salida, )

#     cant_circulos_chicos = len(circulos_chicos)
#     cant_circulos_grandes = len(circulos_grandes)
#     total_circulos = cant_circulos_chicos + cant_circulos_grandes

# plt.figure(figsize=(10, 10))
# plt.imshow(cv.cvtColor(latas_salida, cv.COLOR_BGR2RGB))
# plt.title(f'Total Circles: {total_circulos}, Small: {cant_circulos_chicos}, Large: {cant_circulos_grandes}')
# plt.axis('off')
# plt.show()
