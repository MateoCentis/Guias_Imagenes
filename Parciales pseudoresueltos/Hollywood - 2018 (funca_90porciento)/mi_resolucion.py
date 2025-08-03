import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import segmentar_RGB

escenario0 = cv.imread("Hollywood - 2018 (funca_90porciento)/Hollywood10.jpg")
escenario1 = cv.imread("Hollywood - 2018 (funca_90porciento)/Hollywood11.jpg")
escenario2 = cv.imread("Hollywood - 2018 (funca_90porciento)/Hollywood12.jpg")

# Cambio de color de fondo de la imagen
profesor = cv.imread("Hollywood - 2018 (funca_90porciento)/tito03.jpeg")#(55, 85, 3)

profesor_rgb = cv.cvtColor(profesor, cv.COLOR_BGR2RGB)
profesor_gris = cv.cvtColor(profesor, cv.COLOR_BGR2GRAY)
# _, profesor_bin = cv.threshold(profesor_gris, 50, 255, cv.THRESH_BINARY)
_, profesor_bin = cv.threshold(profesor_gris, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

_,mascara = segmentar_RGB(profesor, 255,0,0,18)#R=255, radio=18 
mascara = cv.bitwise_not(mascara) #shape = (55,85)

color_fondo = [70,135,229]
fondo = np.ones_like(profesor_rgb, dtype=np.uint8)*color_fondo

imagen_mascara = np.where(mascara[..., None] == 255, profesor_rgb, fondo).astype(np.uint8)
profesor_resultado = cv.cvtColor(imagen_mascara, cv.COLOR_RGB2BGR)

escenarios = [escenario0, escenario1, escenario2]

area_min = 14000
area_max = 22000
radio = 20
for escenario in escenarios:
    print("---------------")
    escenario_rgb = cv.cvtColor(escenario, cv.COLOR_BGR2RGB)
    escenario_gris = cv.cvtColor(escenario, cv.COLOR_BGR2GRAY)
    _,escenario_bin = cv.threshold(escenario_gris, 90, 255, cv.THRESH_BINARY)

    contornos, _ = cv.findContours(escenario_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        x, y, w, h = cv.boundingRect(contorno)
        area = w*h
        if area_min <= area <= area_max and w <= 150:
            centro_x = x + int(w/2)
            centro_y = y + int(h/2)
            if np.linalg.norm((escenario_rgb[centro_y,centro_x] - color_fondo),axis=-1) <= radio: 
                min_x = centro_x-profesor_resultado.shape[1]//2
                max_x = centro_x+profesor_resultado.shape[1]//2+1
                min_y = centro_y-profesor_resultado.shape[0]+5
                max_y = centro_y+5
                escenario[min_y:max_y,min_x:max_x,:] = profesor_resultado[:,:,:]

    # Mostrar la imagen con todos los rectángulos dibujados
    plt.imshow(cv.cvtColor(escenario, cv.COLOR_BGR2RGB))
    plt.title('Profesor puesto en lugar faltante')
    plt.show()



