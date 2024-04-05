import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import calcular_histogramas, graficar_histogramas_subplots, mostrar_imagenes

#-Los archivos histo1.tif, histo2.tif, histo3.tif, histo4.tif e histo5.tif
    #contienen histogramas de imagenes con diferentes caracterısticas. Se pide:

ruta1 = "Imagenes_Ej/histo1.tif"
ruta2 = "Imagenes_Ej/histo2.tif"
ruta3 = "Imagenes_Ej/histo3.tif"
ruta4 = "Imagenes_Ej/histo4.tif"
ruta5 = "Imagenes_Ej/histo5.tif"

histo1 = cv.imread(ruta1,cv.IMREAD_GRAYSCALE)
histo2 = cv.imread(ruta2,cv.IMREAD_GRAYSCALE)
histo3 = cv.imread(ruta3,cv.IMREAD_GRAYSCALE)
histo4 = cv.imread(ruta4,cv.IMREAD_GRAYSCALE)
histo5 = cv.imread(ruta5,cv.IMREAD_GRAYSCALE)

#-Analizando solamente los archivos de histograma y realice una descrip-
    #cion de la imagen a la que corresponden (¿es clara u oscura?, ¿tiene buen
    #contraste?, ¿el histograma me explica algo respecto de la ubicacion de
    #los grises?, etc.).
imagenes = [histo1,histo2,histo3,histo4,histo5]


# CONCLUSIÓN: 
    # Imagen 1:
        #Imagen oscura (muchos valores cercanos a 0)
        #
    # Imagen 2:
        #Mucha concentracióin de valores en la zona de los 100
        # Imagen más oscura que clara, promedio < 100
    # Imagen 3: 
        #Mucha concentración en la zona 0-50
        #Brillo medio menor a 30  -> OSCURA
    # Imagen 4: 
        #Imagen clara
    # Imagen 5:
        #Imagen calra

#-Anote la correspondencia histograma-imagen con los archivos imagenA.tif
    #a imagenE.tif, basandose en su analisis previo.

rutaA = "Imagenes_Ej/imagenA.tif"
rutaB = "Imagenes_Ej/imagenB.tif"
rutaC = "Imagenes_Ej/imagenC.tif"
rutaD = "Imagenes_Ej/imagenD.tif"
rutaE = "Imagenes_Ej/imagenE.tif"

imagenA = cv.imread(rutaA,cv.IMREAD_GRAYSCALE)
imagenB = cv.imread(rutaB, cv.IMREAD_GRAYSCALE)
imagenC = cv.imread(rutaC, cv.IMREAD_GRAYSCALE)
imagenD = cv.imread(rutaD,cv.IMREAD_GRAYSCALE)
imagenE = cv.imread(rutaE,cv.IMREAD_GRAYSCALE)

imagenes_originales = [imagenA, imagenB, imagenC, imagenD, imagenE]
#-Cargue las imagenes originales y muestre los histogramas. Comparelos
    #con sus respuestas del punto anterior.

cv.imshow("Imagen A", imagenA)

cv.imshow("Imagen B", imagenB)

cv.imshow("Imagen C", imagenC)

cv.imshow("Imagen D", imagenD)

cv.imshow("Imagen E", imagenE)

mostrar_imagenes(imagenes)

histogramas = calcular_histogramas(imagenes_originales)
graficar_histogramas_subplots(histogramas)

cv.waitKey(0)

# CONCLUSIÓN
    # IMAGEN A: histo2
    # IMAGEN B: histo4
    # IMAGEN D: histo5
    # IMAGEN C: histo1
    # IMAGEN E: histo3

# Ahora vamos a chequear

# VERDAD: 
# -Imagen1 -> histo2 (JOYA)
# -Imagen2 -> histo4 (JOYA)
# -Imagen3 -> histo1 (JOYA)
# -Imagen4 -> histo5 (JOYA)
# -Imagen5 -> histo3 (JOYA)



#-Obtenga y analice la utilidad de las siguientes propiedades estadısticas
    #de los histogramas: media, varianza, asimetrıa, energıa y entropıa.

for histograma in histogramas:
    print(f"Media: {np.mean(histograma)}") #Brillo medio de la imagen
    print(f"Varianza: {np.var(histograma)}")#Contraste
    print(f"Asimetría: {np.std(histograma)}")
    print(f"Energía: {np.sum(histograma)}") #Distribucióin niveles de grises
    print(f"Entropía: {np.log2(np.sum(histograma))} \n") 