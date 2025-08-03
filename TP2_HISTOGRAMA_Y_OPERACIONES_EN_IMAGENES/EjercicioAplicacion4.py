import numpy as np
import cv2 as cv

def ocultar_imagen_binaria(imagen_gris, imagen_binaria):
    # Convertir la imagen binaria a 0s y 1s
    imagen_binaria = imagen_binaria.astype(int)
    
    # Obtener las dimensiones de la imagen binaria
    filas, columnas = imagen_binaria.shape
    
    # Iterar sobre cada píxel de la imagen binaria
    for i in range(filas):
        for j in range(columnas):
            # Obtener el valor del píxel de la imagen en escala de grises
            valor_gris = imagen_gris[i, j]
            # Obtener el bit menos significativo del píxel de la imagen binaria
            bit = imagen_binaria[i, j]
            # Actualizar el bit menos significativo del píxel de la imagen en escala de grises
            valor_gris = (valor_gris & ~1) | bit
            # Asignar el nuevo valor al píxel en la imagen en escala de grises
            imagen_gris[i, j] = valor_gris
    
    return imagen_gris

def extraer_imagen_binaria(imagen_gris):
    # Obtener las dimensiones de la imagen en escala de grises
    filas, columnas = imagen_gris.shape
    
    # Crear una matriz vacía para almacenar la imagen binaria extraída
    imagen_binaria = np.zeros((filas, columnas), dtype=np.uint8)
    
    # Iterar sobre cada píxel de la imagen en escala de grises
    for i in range(filas):
        for j in range(columnas):
            # Obtener el valor del bit menos significativo del píxel
            bit = imagen_gris[i, j] & 1
            # Asignar el valor del bit a la imagen binaria extraída
            imagen_binaria[i, j] = bit
    
    return imagen_binaria

