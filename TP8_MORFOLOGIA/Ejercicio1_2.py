import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
imagen = cv.imread("Morfologia_Im/fosforos.jpg")

imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

umbral = 250
_,imagen_bin = cv.threshold(imagen_gris, umbral, 255, cv.THRESH_BINARY)
imagen_bin = (imagen_bin - 255)

#extraiga en una imagen los fósforos que están verticales y en otra los horizontales.
kernel_vertical = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
],dtype=np.uint8)

verticales_bin = cv.erode(imagen_bin, kernel_vertical)

plt.imshow(verticales_bin, cmap='gray')
plt.show()

kernel_horizontal = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1]])

horizontales_bin = cv.erode(imagen_bin, kernel_horizontal)
plt.imshow(horizontales_bin, cmap='gray')
plt.show()

horizontales = cv.bitwise_and(imagen, imagen, mask=horizontales_bin)
verticales = cv.bitwise_and(imagen, imagen, mask=verticales_bin)

cv.imshow("horizontales",horizontales)
cv.imshow("verticales",verticales)
cv.waitKey(0)


