import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import ventana_trackbars
# 1. Habitualmente las imagenes que se relevan en partes no visibles del espectro
# (como las de infrarrojos, radar, etc.) se encuentran en escala de grises. Para
# resaltar zonas de interes, se pueden asignar colores a rangos especıficos de intensidades.
# Para este ejercicio debe utilizar la imagen ‘rio.jpg’ y resaltar todas las areas
# con acumulaciones grandes de agua (rıo central, ramas mayores y pequeños lagos) en color amarillo.
ruta = "Imagenes_Ej/rio.jpg"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
cv.imshow("RIO",imagen)

# A continuacion le proponemos una guıa metodologica para resolver esto, aunque
    # usted puede proponer otra:
# (a) analizar el histograma y estimar el rango de valores en el que se representa
# el agua
histograma, bins = np.histogram(imagen,bins=256,range=[0,256])
plt.plot(histograma)
plt.show()
cv.destroyAllWindows()
variables_trackbar = ['valor_minimo', 'valor_maximo']
parametros_trackbar = [[0,255],[0,255]]

def transformacion(imagen,valores_trackbar):#EL MEJOR RANGO ES (0,50)
    valor_minimo = valores_trackbar[0]
    valor_maximo = valores_trackbar[1]
    # (b) generar una imagen color cuyos canales son copia de la imagen de intensidad
    # (c) recorrer la imagen original y asignar el color amarillo a los pıxeles cuyas
        # intensidades estan dentro del rango definido
    mascara_agua = cv.inRange(imagen,valor_minimo,valor_maximo)
    imagen_resaltada = cv.cvtColor(imagen, cv.COLOR_GRAY2BGR)
    imagen_resaltada[mascara_agua > 0] = [0,255,255] #Amarillo?
    return imagen_resaltada

# (d) visualizar la imagen resultante y ajustar el rango de grises de ser necesario.
# Consejo: esto se hace mas simple utilizando trackbars.
ventana_trackbars(imagen, variables_trackbar, parametros_trackbar,transformacion)
