import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2
import imutils
from utils import filtro_butterworth_pasa_bajos
#Construir imágenes binarias 
size = 100

# Función para visualizar una imagen binaria
def mostrar_imagen_binaria(image):
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()
def mostrar_imagen_gris(image):
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

# Función para visualizar la TDF
def mostrar_fft(fft_image):
    # np.log(np.abs(fft_image) + 1) #log para comprimir valores, +1 para evitar valores cercanos a 0 y abs para quitar complejos
    # plt.imshow(np.log(np.abs(fft_image) + 1), cmap='gray')
    plt.imshow(np.abs(fft_image) + 1, cmap='gray')
    plt.axis('off')
    plt.colorbar(label='Log Magnitude')
    plt.show()

# Función para calcular la TDF de una imagen binaria
def calcular_fft(binary_image):
    fft_image = fftshift(fft2(binary_image)) #fft2 calcula la transformada 2d y fftshift la centra
    return fft_image

################################################################################################################################
#                                                      Ejercicio 1
################################################################################################################################
# Línea Horizontal
linea_horizontal = np.zeros((size, size))
linea_horizontal[size // 2, :] = 1
linea_horizontal = 1 - linea_horizontal
fft_linea_horizontal = calcular_fft(1-linea_horizontal)


# Línea Vertical
linea_vertical = np.zeros((size, size))
linea_vertical[:, size // 2] = 1
# linea_vertical[:, size // 4 + size//2] = 1
fft_linea_vertical = calcular_fft(linea_vertical)

patron = cv.imread("Imagenes_Ej/patron2.tif", cv.IMREAD_GRAYSCALE)
patron = 1 - patron
patron = cv.rotate(patron, cv.ROTATE_90_CLOCKWISE)
## PArte extra 
# mostrar_imagen_gris(linea_horizontal)
# mostrar_fft(fft_linea_horizontal)
mostrar_imagen_gris(linea_vertical)
mostrar_fft(fft_linea_vertical)
# mostrar_imagen_binaria(patron)
# mostrar_fft(calcular_fft(patron))
# D0 = 40
# n = 5
# sizes = (500,500)
# filtro = filtro_butterworth_pasa_bajos(sizes, D0, n)
# mostrar_imagen_gris(filtro)
# mostrar_fft(calcular_fft(filtro))


# Cuadrado Centrado
cuadrado = np.zeros((size, size))
cuadrado[size // 4: size * 3 // 4, size // 4: size * 3 // 4] = 1
fft_cuadrado = calcular_fft(1-cuadrado)


# Rectángulo Centrado
rectangulo = np.zeros((size, size))
rectangulo[size // 4: size * 3 // 4, size // 6: size * 5 // 6] = 1
fft_rectangulo = calcular_fft(1-rectangulo)


# Círculo (Aproximación)
circulo = np.zeros((size, size))
radio = min(size // 4, size // 6)  # Ajusta el radio para que quepa en la imagen
centro = (size // 2, size // 2)
for i in range(size):
    for j in range(size):
        if (i - centro[0])**2 + (j - centro[1])**2 <= radio**2:
            circulo[i, j] = 1
fft_circulo = calcular_fft(1-circulo)


mostrar_imagenes_binarias = False
if mostrar_imagenes_binarias:
    mostrar_imagen_binaria(1-linea_horizontal)
    mostrar_imagen_binaria(1- linea_vertical)
    mostrar_imagen_binaria(1-cuadrado)
    mostrar_imagen_binaria(1-rectangulo)
    mostrar_imagen_binaria(1-circulo)

################################################################################################################################
#                                                      Ejercicio 2
################################################################################################################################
mostrar_transformadas = False
if mostrar_transformadas:
    mostrar_fft(fft_linea_horizontal)
    mostrar_fft(fft_linea_vertical)
    mostrar_fft(fft_cuadrado)
    mostrar_fft(fft_rectangulo)
    mostrar_fft(fft_circulo)


################################################################################################################################
#                                                      Ejercicio 3
################################################################################################################################
ejer3 = False
if ejer3:
    # Construir la linea vertical
    sizeImagen = 512 
    imagen = np.zeros((sizeImagen, sizeImagen))

    ancho_linea = 1
    centro = sizeImagen // 2

    imagen[:, centro - ancho_linea // 2:centro + ancho_linea // 2 + 1] = 1


    # Rotar la imagen 20 grados
    angulo = 20
    imagen_rotada = imutils.rotate(imagen, angulo)

    #extraer sección de 256x256 de la imagen original y rotada
    sizeSeccion = 256
    fila_inicio = (sizeImagen - sizeSeccion) // 2
    fila_fin = fila_inicio + sizeSeccion
    columna_inicio = (sizeImagen - sizeSeccion) // 2
    columna_fin = columna_inicio + sizeSeccion
    seccion_original = imagen[fila_inicio:fila_fin, columna_inicio:columna_fin]
    seccion_rotada = imagen_rotada[fila_inicio:fila_fin, columna_inicio:columna_fin]

    mostrar_imagen_binaria(1-seccion_original)
    mostrar_imagen_binaria(1-seccion_rotada)

    # Cálculo de la TDF para ambas secciones

    seccion_original_fft = calcular_fft(1-seccion_original)
    seccion_rotada_fft = calcular_fft(1-seccion_rotada)

    mostrar_fft(seccion_original_fft)
    mostrar_fft(seccion_rotada_fft)

################################################################################################################################
#                                                      Ejercicio 4
################################################################################################################################
# Cargar imágenes y visualizar la mangitud de la TDF. Inferir a grandes rasgos la correspondencia entre componentes frecuenciales
    # y detalles de las imágenes
    
#TODO

