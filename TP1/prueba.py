import cv2
import numpy as np
from matplotlib import pyplot as plt
def nothing(x):
  pass

def mostrarGraficoTrackbar(imagen_original):
  NOMBRE_VENTANA = "IMAGEN TRANSFORMADA"
  cv2.namedWindow(NOMBRE_VENTANA)

  cv2.createTrackbar("a", NOMBRE_VENTANA,10,100,nothing)
  cv2.createTrackbar("c", NOMBRE_VENTANA,0,200,nothing)

  # img = cv2.imread('imagenes_varias/micky.jpg',cv2.IMREAD_GRAYSCALE)
  # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

  while(1):
    a = cv2.getTrackbarPos("a", NOMBRE_VENTANA)
    c = cv2.getTrackbarPos("c", NOMBRE_VENTANA)

    #  imagen_salida = np.clip(a * img + c, 0, 255).astype(np.uint8)
    imagen_salida = cv2.convertScaleAbs((a/10)*imagen_original+(c-100))

    cv2.imshow("Imagen resultado",imagen_salida)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
      break
  cv2.destroyAllWindows()