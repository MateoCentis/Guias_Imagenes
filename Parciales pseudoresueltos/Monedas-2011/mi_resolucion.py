import cv2 as cv
import utils as ut

monedas1 = cv.imread('01_Monedas.jpg')
monedas2 = cv.imread('02_Monedas.jpg')

# Contar cantidad de monedas

monedas = [monedas1, monedas2]

for moneda in monedas:
    ut.trackbar_segmentacion_hsv(moneda)
    moneda_gris = cv.cvtColor(moneda, cv.COLOR_BGR2GRAY)
    moneda_bin = cv.threshold(moneda_gris,253,255, cv.THRESH_BINARY)[1]
    moneda_bin = cv.bitwise_not(moneda_bin)
    resultado = ut.encontrar_componentes_y_posiciones(moneda_bin)

    print("Cantidad de monedas: ", resultado["num_componentes"])