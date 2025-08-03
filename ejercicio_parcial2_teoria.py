import cv2 as cv
from utils import reconstruir_imagen, diferencia
import numpy as np
from scipy.ndimage import label, find_objects

def contar_elementos(imagen):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(imagen)
    return num_labels-1



celulas = cv.imread("celulas.jpg", cv.IMREAD_GRAYSCALE)
_,celulas = cv.threshold(celulas, 127, 255, cv.THRESH_BINARY)


# Etiquetar las regiones conectadas
labeled_image, num_features = label(celulas)

sizes = []
for i in range(1, num_features + 1):
    # Obtener la máscara de la región actual
    region_mask = (labeled_image == i)
    # Calcular el área (número de píxeles) de la región
    region_size = np.sum(region_mask)
    sizes.append(region_size)

# Determinar umbrales para clasificar los tamaños de las células
sizes_array = np.array(sizes)
small_threshold = np.percentile(sizes_array, 15)
medium_threshold = np.percentile(sizes_array, 45)

# Contar la cantidad de células de cada tamaño
small_cells = np.sum(sizes_array <= small_threshold)
medium_cells = np.sum((sizes_array > small_threshold) & (sizes_array <= medium_threshold))
large_cells = np.sum(sizes_array > medium_threshold)

print(small_cells, medium_cells, large_cells)



# ANDUVO
solucion_mia = False
if solucion_mia:
    kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ], dtype=np.uint8)

    # Con 10 iteraciones se consiguen todas menos las más pequeñas (guardo)
    celulas_erosionada_grandes_medianas = cv.erode(celulas, kernel, iterations=10)

    # Con 20 solo me quedan las grandes
    celulas_erosionadas_solo_grandes = cv.erode(celulas, kernel, iterations=20)

    celulas_erosionadas_grandes_medianas = reconstruir_imagen(celulas_erosionada_grandes_medianas, celulas)

    celulas_erosionadas_solo_grandes = reconstruir_imagen(celulas_erosionadas_solo_grandes, celulas)
    celulas_erosionadas_solo_medianas = diferencia(celulas_erosionadas_grandes_medianas, celulas_erosionadas_solo_grandes)
    celulas_erosionadas_solo_chicas = diferencia(celulas, celulas_erosionadas_grandes_medianas)

    # Ya las tengo separadas ahora cuento
    cantidad_grandes = contar_elementos(celulas_erosionadas_solo_grandes)

    cantidad_medianas = contar_elementos(celulas_erosionadas_solo_medianas)

    cantidad_chicas = contar_elementos(celulas_erosionadas_solo_chicas)

    print(cantidad_grandes)
    print(cantidad_medianas)
    print(cantidad_chicas)

    cv.imshow("e", celulas)

    cv.waitKey(0)