import cv2 as cv
import numpy as np
from utils import ventana_trackbars, trackbar_transformacion
from icecream import ic
def post_process_lines(lines, image, tolerance=10):
    # Convertir la imagen a escala de grises si es a color
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Diccionario para almacenar las líneas agrupadas por su ángulo y punto medio
    grouped_lines = {}

    # Agrupar las líneas por su ángulo y punto medio
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)  # Calcular el ángulo de la línea
        mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calcular el punto medio de la línea
        key = (angle, mid_point)
        if key not in grouped_lines:
            grouped_lines[key] = []
        grouped_lines[key].append(line)

    # Unir segmentos colineales
    merged_lines = []
    for key, similar_lines in grouped_lines.items():
        if len(similar_lines) > 1:  # Si hay más de una línea colineal
            # Calcular la mediana del ángulo para suavizar las diferencias
            angle = np.median([line[0][2] - line[0][0] for line in similar_lines])
            # Calcular el punto medio promedio
            mid_point = np.mean([np.array(line[0][:2] + line[0][2:]) for line in similar_lines], axis=0).astype(int)
            # Calcular la distancia promedio al punto medio en la imagen original
            avg_distance = np.mean([np.linalg.norm(np.array(line[0][:2]) - np.array(line[0][2:])) for line in similar_lines])
            # Unir los segmentos colineales con una tolerancia de distancia y ángulo
            merged_lines.append((*mid_point, *(mid_point + 1000 * np.array([np.cos(angle), np.sin(angle)]).astype(int)), avg_distance))

    return np.array([[[line[0], line[1], line[2], line[3]]] for line in merged_lines], dtype=np.int32)

# Load image
image_path = 'Imagenes_Ej/camino.tif'
image = cv.imread(image_path)

# # Aplicar Canny edge detection
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize=3)

# # Aplicar la transformada de Hough probabilística
# lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

# # Aplicar el algoritmo de post-procesamiento para unir segmentos colineales
# merged_lines = post_process_lines(lines, image, tolerance=10)
# # Dibujar las líneas unidas en la imagen original
# for line in merged_lines:
#     x1, y1, x2, y2 = line[0]
#     cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

variables_trackbar = ['treshold1', 'treshold2', 'rho','theta','treshold','minLineLength', 'maxLineGap','tolerance']
parametros_trackbar = [[1,255],[1,255], [1,255], [1,360],[1,255], [1,255], [1,255], [1,100]]

def transformacion(imagen, valores_trackbar):
    treshold1 = valores_trackbar[0]
    treshold2 = valores_trackbar[1]
    rho = valores_trackbar[2]
    theta = valores_trackbar[3]*(np.pi/180)
    threshold = valores_trackbar[4]
    minLineLength = valores_trackbar[5]
    maxLineGap = valores_trackbar[6]
    tolerance = valores_trackbar[7]
    image = np.copy(imagen)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, treshold1, treshold2)
    lines = cv.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    print("----------------------------------------")
    if lines is not None:
        merged_lines = post_process_lines(lines, image, tolerance)
        ic(merged_lines)
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # merged_lines = post_process_lines(lines, image, tolerance)
    # for line in merged_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

ventana_trackbars(image,variables_trackbar,parametros_trackbar,transformacion)