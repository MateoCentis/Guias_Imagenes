import cv2 as cv
import numpy as np
from utils import evitar_desborde, mostrar_imagenes
# Ejercicio 3: Filtros de acentuado
# 1. Obtenga versiones mejoradas de diferentes imagenes mediante el ﬁltrado por
# mascara difusa. Implemente el calculo como
# g(x, y) = f (x, y) − P B(f (x, y))
# 2. Una forma de enfatizar las altas frecuencias sin perder los detalles de bajas
# frecuencias es el ﬁltrado de alta potencia. Implemente este procesamiento
# como la operacion aritmetica:
# g(x, y) = Af (x, y) − P B(f (x, y)), con A ≥ 1.
# * Investigue y pruebe metodos alternativos de calculo en una pasada
def acentuar(imagen,sizeMascara,A):
    mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    imagen_promedio = cv.filter2D(imagen, -1, mascara_promedio)
    
    mascara_cruz = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)
    imagen_cruz = cv.filter2D(imagen, -1, mascara_cruz)
    
    
    imagen_difusa = cv.GaussianBlur(imagen,(sizeMascara,sizeMascara),0)
    
    imagen_salida1 = evitar_desborde(A*imagen - imagen_difusa)
    imagen_salida2 = evitar_desborde(A*imagen - imagen_promedio)
    imagen_salida3 = evitar_desborde(A*imagen - imagen_cruz)
    return [imagen_salida1,imagen_salida2,imagen_salida3]

ruta = "Imagenes_Ej/building.jpg"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
sizeMascara = 3
A = 1.5
imagenes_salida = acentuar(imagen,sizeMascara,A)
mostrar_imagenes([imagen,imagenes_salida[0],imagenes_salida[1],imagenes_salida[2]])


