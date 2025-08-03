import cv2 as cv
import numpy as np

tablero = cv.imread("Ajedrez - 2021/B00.png")


cv.imshow("d",tablero)
cv.waitKey(0)

# Dama: 9 puntos;
# Torre: 5 puntos;
# Alfil: 3 puntos;
# Caballo: 3 puntos;
# Pe´on: 1 punto