import cv2 as cv
import numpy as np
from utils import mostrar_imagenes, calcular_histogramas, graficar_histogramas_subplots
# 1. En la imagen cuadros.tif se observa un conjunto de cuadros negros sobre un
# fondo casi uniforme.
ruta = "Imagenes_Ej/cuadros.tif"
cuadros = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

# Utilice ecualizacion local del histograma para revelar los
# detalles ocultos en la imagen y compare los resultados con los obtenidos con
# ecualizacion global.

cuadros_ecualizados_global = cv.equalizeHist(cuadros)

clahe = cv.createCLAHE(clipLimit=15.0, tileGridSize=(15,15))
# Ayuda: el tamaño de ventana y su localizacion es clave para realizar la ecualizacion local.
cuadros_ecualizados_local = clahe.apply(cuadros)

# clipLimit:
    # Este parámetro controla la cantidad de recorte aplicado al histograma.
    # Un valor más alto de clipLimit permite un mayor contraste local, pero también puede aumentar el ruido.
    # Si clipLimit es demasiado bajo, la ecualización local no tendrá un efecto significativo.
    # Ajusta este valor según tus necesidades y el tipo de imagen.
# tileGridSize:
    # Define el tamaño de los mosaicos (ventanas) en los que se divide la imagen.
    # Cada mosaico se ecualiza de forma independiente.
    # Un tamaño de mosaico más pequeño captura detalles finos, pero puede introducir ruido.
    # Un tamaño de mosaico más grande suaviza la imagen, pero puede perder detalles locales.
    # El valor (8, 8) significa que la imagen se divide en mosaicos de 8x8 píxeles.

mostrar_imagenes([cuadros_ecualizados_global,cuadros_ecualizados_local])

histogramas = calcular_histogramas([cuadros_ecualizados_global,cuadros_ecualizados_local])
graficar_histogramas_subplots(histogramas)