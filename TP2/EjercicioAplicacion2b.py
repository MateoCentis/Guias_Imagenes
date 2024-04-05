import cv2 as cv
import numpy as np
from utils import mostrar_imagenes, graficar_histogramas_subplots, calcular_histogramas

ruta_base = "Imagenes_Ej/Busqueda_histograma/"
GRIS = cv.IMREAD_GRAYSCALE
# Cargamos las imágenes
personajes = []
caricaturas = []
banderas = []
paisajes = []
for i in range (5):
    ruta_personaje = "Personaje0"+str(i+1)+".jpg"
    ruta_bandera = "Bandera0"+str(i+1)+".jpg"
    ruta_paisaje = "Paisaje0"+str(i+1)+".jpg"
    ruta_caricatura = "Caricaturas0"+str(i+1)+".jpg"

    personajes.append(cv.imread(ruta_base + ruta_personaje, GRIS))
    banderas.append(cv.imread(ruta_base + ruta_bandera, GRIS))
    paisajes.append(cv.imread(ruta_base + ruta_paisaje, GRIS))
    caricaturas.append(cv.imread(ruta_base + ruta_caricatura, GRIS))

# 2. Realice un algoritmo de busqueda por correlacion de histogramas de intensi-
# dad. Se debe informar el contenido de la imagen: Bandera, Caricatura, Perso-
# naje o Paisaje. Utilice las imagenes disponibles en Busqueda histograma.zip.

#1. Cálculo de histogramas
histogramas_personajes = calcular_histogramas(personajes)

histogramas_banderas = calcular_histogramas(banderas)

histogramas_paisajes = calcular_histogramas(paisajes)

histogramas_caricaturas = calcular_histogramas(caricaturas)

mostrar_imagenes(personajes)


#Función que dada un histograma y una lista de histogramas de referencia 
    #te devuelve el valor máximo de correlacion
def maxima_correlacion_histograma(histograma, histogramas_referencia):
    correlaciones = []
    for hist_ref in histogramas_referencia:
        histograma_float = cv.normalize(histograma, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        hist_ref_float = cv.normalize(hist_ref, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        correlacion = cv.compareHist(histograma_float, hist_ref_float, cv.HISTCMP_CORREL)
        correlaciones.append(correlacion)

    return np.max(correlaciones)

imagen_ejemplo = personajes[1]
histograma_ejemplo = histogramas_personajes[1]

maximas_correlaciones = []
histogramas = [histogramas_banderas,histogramas_caricaturas,histogramas_personajes,histogramas_paisajes]
#Recorro cada una y chequeo
for i in range(4):
    maximas_correlaciones.append(maxima_correlacion_histograma(histograma_ejemplo,histogramas[i]))

indice_maximo = np.argmax(maximas_correlaciones)
if indice_maximo == 0:
    print("La imagen es de tipo: BANDERA")
elif indice_maximo == 1:
    print("La imagen es de tipo: CARICATURA")
elif indice_maximo == 2:
    print("La imagen es de tipo: PERSONAJE")
else:
    print("La imagen es de tipo: PAISAJE")

