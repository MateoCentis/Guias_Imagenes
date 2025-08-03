import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#are collinear comprueba si dos linas con colineales y están dentro de una distancia de tolerancia
def are_collinear(line1, line2, angle_tolerance, distance_tolerance):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    def angle(x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1)

    def distance_point_to_line(x0, y0, x1, y1, x2, y2):
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    angle1 = angle(x1, y1, x2, y2)
    angle2 = angle(x3, y3, x4, y4)
    
    angle_diff = abs(angle1 - angle2)
    
    if angle_diff < angle_tolerance or abs(angle_diff - np.pi) < angle_tolerance:
        d1 = distance_point_to_line(x1, y1, x3, y3, x4, y4)
        d2 = distance_point_to_line(x2, y2, x3, y3, x4, y4)
        
        if d1 < distance_tolerance and d2 < distance_tolerance:
            return True
    
    return False

#Une líneas colineales en un solo segmetno
def merge_lines(lines, angle_tolerance, distance_tolerance):
    merged_lines = []

    while len(lines) > 0:
        line = lines.pop(0)
        x1, y1, x2, y2 = line
        merged = False

        for i, (mx1, my1, mx2, my2) in enumerate(merged_lines):
            if are_collinear(line, [mx1, my1, mx2, my2], angle_tolerance, distance_tolerance):
                merged_lines[i] = [min(x1, mx1, mx2, x2), min(y1, my1, my2, y2), max(x1, mx1, mx2, x2), max(y1, my1, my2, y2)]
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    return merged_lines

#Aplica Hough probabilística y post-procesa para unir lineas 
def apply_hough_lines_p_with_postprocess(image_path, rho, theta, threshold, min_line_length, max_line_gap, angle_tolerance, distance_tolerance):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        lines = [line[0] for line in lines]
        merged_lines = merge_lines(lines, angle_tolerance, distance_tolerance)
    else:
        merged_lines = []

    line_image = np.copy(image)
    for x1, y1, x2, y2 in merged_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image, edges, line_image

def display_images(original, edges, lines):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Edges')
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Hough Lines with Post-processing')
    plt.imshow(cv2.cvtColor(lines, cv2.COLOR_BGR2RGB))

    plt.show()

# Parámetros
rho = 1
theta = np.pi / 180 
threshold = 100
min_line_length = 10
max_line_gap = 50
angle_tolerance = np.pi / 180 * 2  # Tolerancia angular de 5 grados
distance_tolerance = 10  # Tolerancia de distancia de 10 píxeles

# Aplicar a las imágenes
image_files = ['Imagenes_Ej/letras1.tif', 'Imagenes_Ej/letras2.tif', 'Imagenes_Ej/snowman.png', 'Imagenes_Ej/building.jpg']
for image_file in image_files:
    original, edges, lines = apply_hough_lines_p_with_postprocess(image_file, rho, theta, threshold, min_line_length, max_line_gap, angle_tolerance, distance_tolerance)
    display_images(original, edges, lines)
