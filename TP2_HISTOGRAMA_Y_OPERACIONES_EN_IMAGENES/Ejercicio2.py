import cv2 as cv
import numpy as np

def nothing(x):
    pass

def logaritmica(r):
    global c 
    return c*np.log(1 + r)

def potencia(r, gamma):
    global c
    return c*np.power(r, gamma)

ruta = "Imagenes_Ej/rmn.jpg"
imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)

NOMBRE_VENTANA = "Logaritmica y potencia"
cv.namedWindow(NOMBRE_VENTANA)

cv.createTrackbar("gamma", NOMBRE_VENTANA, 10, 100, nothing)
cv.createTrackbar("c",NOMBRE_VENTANA,0,255,nothing)
while True:
    gamma = cv.getTrackbarPos("gamma", NOMBRE_VENTANA) / 20
    c = cv.getTrackbarPos("c",NOMBRE_VENTANA)
    # Aplicar la transformación logarítmica y de potencia
    imagen_salida_logaritmica = np.clip(logaritmica(imagen), 0, 255).astype(np.uint8)
    imagen_salida_potencia = np.clip(potencia(imagen, gamma), 0, 255).astype(np.uint8)

    # Concatenar imágenes horizontalmente
    imagen_resultado = np.hstack((imagen_salida_logaritmica, imagen_salida_potencia))

    cv.imshow(NOMBRE_VENTANA, imagen_resultado)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
