import cv2 as cv
import numpy as np
from utils import ventana_trackbars
import matplotlib.pyplot as plt
# Ejercicio 2: Filtros pasa-altos
# 1. Deﬁna mascaras de ﬁltrado pasa-altos cuyos coeﬁcientes sumen 1 y aplıquelas
# sobre diferentes imagenes. Interprete los resultados.
# ruta = "Imagenes_Ej/mosquito.jpg"
# imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

ruta = "Imagenes_Ej/flores02.jpg"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

m1 = np.array([[0.25,0.5,0.25]])
m2 = np.array([[0.25],[0.5],[0.25]])

imagen_transformada1 = cv.filter2D(imagen, -1, m1)
imagen_transformada2 = cv.filter2D(imagen, -1, m2)

plt.imshow(imagen_transformada1, cmap='gray')
plt.show()
plt.imshow(imagen_transformada2, cmap='gray')
plt.show()

m12 = m2@m1 

imagen_transformada12 = cv.filter2D(imagen, -1, m12)

plt.imshow(imagen_transformada12, cmap='gray')

plt.show()


ver_mascaras = False
if ver_mascaras:
    mascara1 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    mascara2 = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    mascara3 = np.array([[1, -2, 1],
                        [-2, 5, -2],
                        [1, -2, 1]])
    variables_trackbar = ["NumeroMascara"]
    parametros_trackbar = [[1,4]]

    def transformacion(imagen,valores_trackbar):
        valor_mascara = valores_trackbar[0]
        global mascara1, mascara2, mascara3
        if valor_mascara < 1:
            valor_mascara = 1
        if valor_mascara > 3:
            valor_mascara = 3

        if valor_mascara == 1:
            mascara = mascara1
        if valor_mascara == 2:
            mascara = mascara2
        if valor_mascara == 3:
            mascara = mascara3
        
        imagen_transformada = cv.filter2D(imagen, -1, mascara)

        return imagen_transformada


    ventana1 = ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)

    # 2. Repita el ejercicio anterior para mascaras cuyos coeﬁcientes sumen 0. Com-
    # pare los resultados con los del punto anterior

    # cambio los valores de valor_mascara y listo
    mascara1 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    mascara2 = np.array([[-1, -2, -1],
                        [-2, 12, -2],
                        [-1, -2, -1]])
    mascara3 = np.array([[0, -2, 0],
                        [-2, 8, -2],
                        [0, -2, 0]])

    ventana2 = ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)
