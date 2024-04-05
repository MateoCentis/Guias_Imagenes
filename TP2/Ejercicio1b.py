import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Cargamos las imágenes
ruta1 = "Imagenes_Ej/patron2.tif"
ruta2 = "Imagenes_Ej/patron.tif"

patron2 = cv.imread(ruta1)
patron = cv.imread(ruta2, cv.IMREAD_GRAYSCALE)

# Calculamos los histogramas
#Función calcHist PARÁMETROS:
    #images: lista de imagenes representadas como matrices numpy 
        #(todas las imágenes deben tener mismo tamaño y dtype)
    #channels: lista de canales (si RGB calcula para cada canal)
        #[0] es por defecto
    #mask: Matriz de 8 bits del mismo tamaño que la imagen
        #se usa para calcular el histograma a solo una parte de la imagen
    #histSize: tamaño del histograma en cada dirección
    #ranges: array de arrays de dimensiones que define los límites 
        #Por lo general [0,256], para escala de grises [0,256,0,256,0,256]
    #hist: histograma de salida, array que contiene la frecuencia de ocurrencia
        # de los valores de intensidad
hist_patron2 = cv.calcHist([patron2], [0], None, [256], [0, 256])
hist_patron = cv.calcHist([patron], [0], None, [256], [0, 256])

# Graficamos los histogramas
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist_patron2, color='b')
plt.title("Histograma de 'patron2.tif'")
plt.xlabel("Intensidad de píxeles")
plt.ylabel("Frecuencia")

plt.subplot(2, 1, 2)
plt.plot(hist_patron, color='r')
plt.title("Histograma de 'patron.tif'")
plt.xlabel("Intensidad de píxeles")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()
