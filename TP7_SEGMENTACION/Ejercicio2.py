import numpy as np
import cv2 as cv
from utils import trackbar_transformacion, ventana_trackbars
from icecream import ic
"""Antes de comenzar, recuerde ¿que tipo de imagenes son apropiadas para utilizar con TH?, 
    RTA: 
¿que preprocesos serian utiles?, ¿que se obtiene (en el espacio transformado) al aplicar TH a un punto?,
    RTA: 
 ¿que particularidad presentan (en el espacio transformado) los puntos colineales?, ¿que espera y que no, 
 como resultado del proceso de la TH?
    RTA: 
 """

"""Estudie los parametros de la funcion cv.HoughLines, y los formatos en los que 
    devuelve el resultado ([rho, θ] o [rho, θ, votes]). Implemente un algoritmo que haga   
    uso de esta funcion, debe permitir ajustar el rango de los angulos en la busqueda
     de puntos colineales y el umbral para el acumulador. Consejo: use trackbars."""

#---------------------------------------------Parte uno-------------------------------------------------
parte1 = False
if parte1:
    imagen = cv.imread("Imagenes_Ej/camino.tif", cv.IMREAD_GRAYSCALE)
    variables_trackbar : list[str]= ['rho','theta','treshold']
    parametros_trackbar : list[list] = [[1,500], [1,360],[1,255]]

    def transformacion(imagen, valores_trackbar):
        rho = valores_trackbar[0]
        theta = valores_trackbar[1]
        threshold = valores_trackbar[2]
        salida = cv.HoughLines(imagen,rho,theta, threshold)
        print("--------------------------SALIDA--------------------------------")
        ic(rho,theta,threshold)
        ic(salida)
        aux = np.zeros_like(imagen)
        return aux

    ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)
    # Devuelve una matriz de forma (N, 1, 2), donde N es el número de líneas detectadas.
        # Cada línea está representada por un par de valores [rho, theta].
        #votes se refiere al número de votos acumulados para una línea detectada en la transformada de Hough. 
            #Este valor representa cuántas veces se ha intersectado un conjunto de puntos en la imagen original 
            #durante el proceso de detección de líneas.

#---------------------------------------------Parte 2-------------------------------------------------

"""2. Estudie la implementacion de la TH probabilistica cv.HoughLinesP, sus parametros y 
    el formato vectorial en el que devuelve los resultados. Implemente un algoritmo que haga 
    uso de esta funcion, debe permitir ajustar los parametros (minLineLength,maxLineGap) y 
    el umbral para el acumulador. Utilice ambas implementaciones con las imagenes letras1.tif, 
    letras2.tif, snowman.png y building.jpg. Explique sus diferencias, ventajas y desventajas. 
    }¿Cuando utilizaria uno y cuando el otro?"""

def apply_hough_lines_p(image, rho, theta, threshold, min_line_length, max_line_gap):

    edges = cv.Canny(image, 50, 150, apertureSize=3)

    # Aplicar la Transformada de Hough probabilística
    lines = cv.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Crear una imagen para dibujar las líneas
    line_image = np.copy(image)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, edges, line_image

parte2 = True
if parte2:
    letras1 = cv.imread("Imagenes_Ej/letras1.tif", cv.IMREAD_GRAYSCALE)
    letras2 = cv.imread("Imagenes_Ej/letras2.tif", cv.IMREAD_GRAYSCALE)
    snowman = cv.imread("Imagenes_Ej/snowman.png", cv.IMREAD_GRAYSCALE)
    building = cv.imread("Imagenes_Ej/building.jpg", cv.IMREAD_GRAYSCALE)

    variables_trackbar : list[str]= ['rho','theta','treshold', 'minLineLength','maxLineGap']
    parametros_trackbar : list[list] = [[1,500], [1,360],[1,255], [0,255], [0,255]]
    
    def transformacion(imagen, valores_trackbar):
        rho = valores_trackbar[0]
        theta = valores_trackbar[1]
        threshold = valores_trackbar[2]
        minLineLength = valores_trackbar[3]
        maxLineGap = valores_trackbar[4]

        salida, bordes, salida_lineas = apply_hough_lines_p(imagen,rho,theta,threshold,minLineLength, maxLineGap)
        ic(bordes)
        return salida_lineas
    trackbar_transformacion(letras1,variables_trackbar, parametros_trackbar, transformacion)

    #devuelve segmentos de línea definidos por sus puntos finales (x1, y1) y (x2, y2)
    #minLineLength: La longitud mínima de los segmentos de línea que se deben detectar. 
        #Cualquier segmento de línea más corto que este valor será descartado.
    #maxLineGap: La distancia máxima entre dos puntos para que se consideren parte del mismo segmento de línea.

"""
RTA:
Las diferencias entre la transformada de Hough clásica y la transformada de Hough probabilística radican 
en cómo manejan la detección de líneas. La transformada de Hough clásica es más precisa pero computacionalmente
 más costosa ya que considera todas las posibles líneas en la imagen. La transformada de Hough probabilística 
 es más eficiente, ya que examina solo una submuestra de puntos de la imagen, lo que la hace más rápida,
   pero potencialmente menos precisa, especialmente en presencia de ruido o líneas superpuestas.

Para decidir cuál usar, debes considerar el equilibrio entre precisión y eficiencia en tu aplicación específica.
 Si la precisión es crítica y puedes permitirte el costo computacional, la transformada de Hough clásica
puede ser más adecuada. Si la eficiencia es más importante y puedes tolerar una pequeña disminución en
la precisión, entonces la transformada de Hough probabilística puede ser preferible.
 En general, la transformada de Hough probabilística se usa comúnmente en aplicaciones en tiempo real
   donde se requiere una detección rápida de líneas.
"""

