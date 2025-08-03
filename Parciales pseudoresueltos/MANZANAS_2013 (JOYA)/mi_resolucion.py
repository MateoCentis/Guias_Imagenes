import cv2 as cv
import numpy as np
from utils import trackbar_segmentacion_hsv, segmentacion_hsv, reconstruir_imagen, diferencia

# El ejercicio tiene dos partes
    #Primero, contar cuantas manzanas rojas y cuantas verdes
    #Segundo, pintar de azul a las manzanas pequeñas
arbol1 = cv.imread("MANZANAS_2013 (JOYA)/img1.jpg")
arbol2 = cv.imread("MANZANAS_2013 (JOYA)/img2.jpg")

# trackbar_segmentacion_hsv(arbol2)

rango_hue_rojas = (75,360)
rango_hue_verdes = (36,43)
rango_saturation = (210,255)

### Primero: Contar cuantas rojas y cuantas verdes -------------------------------------------------------
segmentacion1_rojas = segmentacion_hsv(arbol1, rango_hue_rojas, rango_saturation)
segmentacion1_verdes = segmentacion_hsv(arbol1, rango_hue_verdes, rango_saturation)
segmentacion2_rojas = segmentacion_hsv(arbol2, rango_hue_rojas, rango_saturation)
segmentacion2_verdes = segmentacion_hsv(arbol2, rango_hue_verdes, rango_saturation)

segmentacion1_rojas_gris = cv.cvtColor(segmentacion1_rojas, cv.COLOR_BGR2GRAY)
segmentacion1_verdes_gris = cv.cvtColor(segmentacion1_verdes, cv.COLOR_BGR2GRAY)
segmentacion2_rojas_gris = cv.cvtColor(segmentacion2_rojas, cv.COLOR_BGR2GRAY)
segmentacion2_verdes_gris = cv.cvtColor(segmentacion2_verdes, cv.COLOR_BGR2GRAY)

segmentacion1_rojas_bin = cv.threshold(segmentacion1_rojas_gris, 40, 255, cv.THRESH_BINARY)[1]
segmentacion1_verdes_bin = cv.threshold(segmentacion1_verdes_gris, 40, 255, cv.THRESH_BINARY)[1]
segmentacion2_rojas_bin = cv.threshold(segmentacion2_rojas_gris, 40, 255, cv.THRESH_BINARY)[1]
segmentacion2_verdes_bin = cv.threshold(segmentacion2_verdes_gris, 40, 255, cv.THRESH_BINARY)[1]

# Erosión para limpiar un poco la imagen así le pasamos el connected components para contar
kernel = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=np.uint8)

segmentacion1_rojas_bin = cv.erode(segmentacion1_rojas_bin,kernel=kernel, iterations=1)
segmentacion1_verdes_bin = cv.erode(segmentacion1_verdes_bin,kernel=kernel, iterations=1)
segmentacion2_rojas_bin = cv.erode(segmentacion2_rojas_bin,kernel=kernel, iterations=1)
segmentacion2_verdes_bin = cv.erode(segmentacion2_verdes_bin,kernel=kernel, iterations=1)

cantidad_rojas1,_ = cv.connectedComponents(segmentacion1_rojas_bin)
cantidad_verdes1,_ = cv.connectedComponents(segmentacion1_verdes_bin)
cantidad_rojas2,_ = cv.connectedComponents(segmentacion2_rojas_bin)
cantidad_verdes2,_ = cv.connectedComponents(segmentacion2_verdes_bin)

print("En la imagen 1 hay de manzanas rojas: ", cantidad_rojas1-1)
print("En la imagen 2 hay de manzanas rojas: ", cantidad_rojas2-1)
print("En la imagen 1 hay de manzanas verdes: ", cantidad_verdes1-1)
print("En la imagen 2 hay de manzanas verdes: ", cantidad_verdes2-1)


### Segundo: pintar de azul las más chicas -----------------------------------------------------------
segmentacion1 = segmentacion_hsv(arbol1, (30,186), rango_saturation)
segmentacion2 = segmentacion_hsv(arbol2, (30,186), rango_saturation)

segmentacion1_gris = cv.cvtColor(segmentacion1, cv.COLOR_BGR2GRAY)
segmentacion2_gris = cv.cvtColor(segmentacion2, cv.COLOR_BGR2GRAY)

segmentacion1_bin = cv.threshold(segmentacion1_gris, 40, 255, cv.THRESH_BINARY)[1]
segmentacion2_bin = cv.threshold(segmentacion2_gris, 40, 255, cv.THRESH_BINARY)[1]

segmentacion1_bin = cv.erode(segmentacion1_bin,kernel=kernel, iterations=1)
segmentacion1_bin = cv.dilate(segmentacion1_bin, kernel=kernel, iterations=1)
segmentacion2_bin = cv.erode(segmentacion2_bin,kernel=kernel, iterations=1)
segmentacion2_bin = cv.dilate(segmentacion2_bin, kernel=kernel, iterations=1)

# Ahora erosionar para obtener imagen sin chiquitas y luego hacer diferencia
segmentacion1_bin_sinchicas = cv.erode(segmentacion1_bin, kernel=kernel, iterations=8)
segmentacion1_bin_sinchicas = reconstruir_imagen(segmentacion1_bin_sinchicas, segmentacion1_bin)
segmentacion1_chicas = diferencia(segmentacion1_bin, segmentacion1_bin_sinchicas)
segmentacion1_chicas = cv.dilate(segmentacion1_chicas, kernel=kernel, iterations=2)

segmentacion2_bin_sinchicas = cv.erode(segmentacion2_bin, kernel=kernel, iterations=8)
segmentacion2_bin_sinchicas = reconstruir_imagen(segmentacion2_bin_sinchicas, segmentacion2_bin)
segmentacion2_chicas = diferencia(segmentacion2_bin, segmentacion2_bin_sinchicas)
segmentacion2_chicas = cv.dilate(segmentacion2_chicas, kernel=kernel, iterations=2)

arbol1_chicas_pintadas = arbol1.copy()
arbol2_chicas_pintadas = arbol2.copy()

# Ahora hay que pintar de azul las chicas
arbol1_chicas_pintadas[segmentacion1_chicas == 255] = [255,0,0]
arbol2_chicas_pintadas[segmentacion2_chicas == 255] = [255,0,0]

cv.imshow("Arbol 1 - chicas pintadas de azul", arbol1_chicas_pintadas)
cv.imshow("Arbol 2 - chicas pintadas de azul", arbol2_chicas_pintadas)

cv.waitKey(0)


# cv.imshow("seg", segmentacion)
# cv.waitKey(0)
