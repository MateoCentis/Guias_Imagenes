import cv2 as cv
import numpy as np
from utils import mostrar_imagenes
from utils import mostrar_grafico_trackbar
##PARÉNTESIS: lo de la lut es hacer los 256 valores posibles que pueden tomar los grises transformados entonces 
    #a la hora de hacer operaciones solo tomo uno de los valores posibles calculados, sobrán recursos así que una paja
#Transformaciones lineales de una imagen
#s = a*r + c ; r: valor de entrada, a: ganancia, c: offset

ruta = "imagenes_varias/micky.jpg"
imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)

# 1. Implementa una LUT del mapeo entre la entrada y la salida
# LUT: vector donde el índice con el que se accede a el es r, y el valor almacenado será s
# Generamos un vector que tiene 0-255 elementos donde cada elemento adentro va a tener el valor s almacenado
    #a partir de los valores a y c definidos (ir probando diferentes valores)
a = 2
c = 0

entrada = np.arange(256)
salida = np.clip(a * entrada + c, 0, 255).astype(np.uint8)  # Aplicar la transformación y limitar los valores a [0, 255]

# Aplicar la LUT a todos los valores posibles de entrada
imagen_transformada = salida[imagen]

mostrar_imagenes([imagen, imagen_transformada])

# 2. Pruebee la rutina con diferentes coeficientes, sobre varias imagenes, muestre en una misma ventana 
    #imagen original, mapeo aplicado e imagen obtenida

mostrar_grafico_trackbar(imagen)
# mostrar_imagenes([imagen,imagen_transformada])
print(f"Mapeo: s = {a}*r + {c}")

# 3. Implemente el negativo

def negativo(imagen):
    return 255 - imagen

imagen_negativo = negativo(imagen)
mostrar_imagenes([imagen,imagen_negativo])

# 4. Genere diversas LUT con estiramientos y compresiones lineales por tramos de la entrada, y pruebe los resultados sobre diversas imágenes

# Función para generar una LUT con estiramiento o compresión lineal por tramos (??)
def generar_lut_por_tramos(tramos):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(len(tramos) - 1):
        rango_inicial, rango_final = tramos[i], tramos[i + 1]
        a = 255 / (rango_final - rango_inicial)
        c = -a * rango_inicial
        for r in range(rango_inicial, rango_final):
            lut[r] = np.clip(a * r + c, 0, 255)
    return lut

# Definir los tramos de la transformación lineal por segmentos
tramos = [0, 50, 100, 150, 200, 255]

# Generar la LUT
lut = generar_lut_por_tramos(tramos)

# Aplicar la LUT a una imagen de ejemplo
# imagen = cv.imread('ejemplo.jpg', cv.IMREAD_GRAYSCALE)
imagen_transformada = cv.LUT(imagen, lut)

# Mostrar imagen original y transformada
mostrar_imagenes([imagen, imagen_transformada])
