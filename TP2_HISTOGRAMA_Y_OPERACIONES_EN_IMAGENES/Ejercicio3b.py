import cv2 as cv
import numpy as np
from utils import calcular_histogramas, graficar_histogramas_subplots, mostrar_imagenes
# Cargue una imagen y realice la ecualizacion de su histograma.
ruta = "Imagenes_Ej/imagenA.tif"
imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
imagen_ecualizada = cv.equalizeHist(imagen)

histogramas = calcular_histogramas([imagen,imagen_ecualizada])

mostrar_imagenes([imagen,imagen_ecualizada])
graficar_histogramas_subplots(histogramas)

#Muestre en una misma ventana la imagen original, la version ecualizada y sus respectivos histogramas.
# Estudie la informacion suministrada por los histogramas. 
    #¿Que diferencias nota respecto a las deﬁniciones teoricas?
# Repita el analisis para distintas imagenes.
    
#CONCLUSIÓN: Se puede ver que la ecualizada tiene un poco más distribuidos los niveles de grises