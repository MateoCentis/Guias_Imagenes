import cv2 as cv
import numpy as np

# - El logo FICH debe ir 1 o 2 pixeles debajo de la linea horizontal y 1 o 2 pixeles
    # a la izquierda de la linea vertical.
# - El logo UNL debe ir 1 o 2 pixeles debajo de la linea horizontal y 1 o 2 pixeles
    # a la derecha de la linea vertical.
# - El logo sinc(i) debe ir centrado respecto de la linea vertical y 1 o 2 pixeles
    # por encima del borde inferior de la imagen.

# Dado que cada partido se desarrolla en un estadio diferente, el campo de juego
    # puede variar de color

# Para que el efecto resulte creible a los espectadores, los jugadores tienen que apa-
#     recer desplazandose sobre los carteles publicitarios, como si este verdaderamente
#     estuviera pintado sobre la superficie de juego.


cancha1 = cv.imread("CANCHAPUBICA_2016/1.png")
cancha2 = cv.imread("CANCHAPUBICA_2016/2.png")
cancha3 = cv.imread("CANCHAPUBICA_2016/3.png")
