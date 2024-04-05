import cv2 as cv
import numpy as np
from utils import mostrar_imagenes
import matplotlib.pyplot as plt
# 2. Aplique un filtro pasa-bajos de su elección y el filtro bilateral a las siguientes
# imagenes: mariposa02.png, flores02.jpg y lapices02.jpg (en escala de
# grises).
ruta1 = "Imagenes_Ej/mariposa02.png"
ruta2 = "Imagenes_Ej/flores02.jpg"
ruta3 = "Imagenes_Ej/lapices02.jpg"

mariposa = cv.imread(ruta1, cv.IMREAD_GRAYSCALE)
flores = cv.imread(ruta2, cv.IMREAD_GRAYSCALE)
lapices = cv.imread(ruta3, cv.IMREAD_GRAYSCALE)

kernel_size = 3
mariposa_PB = cv.blur(mariposa,(kernel_size,kernel_size))
flores_PB = cv.blur(flores,(kernel_size,kernel_size))
lapices_PB = cv.blur(lapices,(kernel_size,kernel_size))

diametro = 9 #Diámetro del vecindario alrededor del píxel que se utilizará durante el cálculo. 
                #Un valor más grande de d significa que se consideran más píxeles en el vecindario.
sigma_color = 75#Un parámetro que controla cuánta diferencia de color se permitirá entre los píxeles 
                #dentro del vecindario. Un valor más grande de sigmaColor significa que se considerará 
                    #una mayor diferencia de color (en este caso intensidades?).
sigma_space = 75 # Parámetro que controla cuánta diferencia de posición espacial se permitirá entre los 
                #píxeles dentro del vecindario. Un valor más grande de sigmaSpace significa que se 
                    #considerarán píxeles más alejados en el espacio.
mariposa_BI = cv.bilateralFilter(mariposa,d=diametro, sigmaColor=sigma_color, sigmaSpace=sigma_space)
flores_BI = cv.bilateralFilter(flores,d=diametro,sigmaColor=sigma_color, sigmaSpace=sigma_space)
lapices_BI = cv.bilateralFilter(lapices,d=diametro, sigmaColor=sigma_color, sigmaSpace= sigma_space)

# Compare los resultados y explique sus apreciaciones.
mostrar = False
if mostrar:
    mostrar_imagenes([mariposa,mariposa_PB,mariposa_BI])
    mostrar_imagenes([flores,flores_PB,flores_BI])
    mostrar_imagenes([lapices,lapices_PB,lapices_BI])

# Utilice la función implementada en la guıa anterior para visualizar perfiles
# de grises, eligiendo la misma fila o columna para la imagen origina y las que han sido filtradas. 
    #Compare los resultados visualizandolos simultaneamente.

# [Opcional] Implemente una función que le permita extraer perfiles de
# grises de las 3 imagenes, de cualquier longitud y en cualquier direccion
# (a partir de clicks del mouse o mediante el ingreso de coordenadas) y que
# realice el ploteo de los perfiles superpuestos en diferentes colores.
# 3


def extraer_perfiles(imagen1, imagen2, imagen3):
    def on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print("X: ", x)
            print("Y: ", y)
            perfilesX = []
            perfilesY = []
            imagenes = [imagen1,imagen2,imagen3]
            #Perfil de intensidad filas
            for i in range(len(imagenes)):
                perfilesX.append(imagenes[i][x,:])
            plt.figure(figsize=(10, 5))
            colors = ['blue', 'red', 'green']
            labels = ['Original', 'Filtro PasaBajos', 'Filtro Bilateral']
            for i, profile in enumerate(perfilesX):
                plt.plot(perfilesX[i], color=colors[i], label=labels[i])
            plt.title(f'Perfiles de Grises de intesidad de la fila {x}')
            plt.grid(True,'minor')
            plt.xlabel('y')
            plt.ylabel('Intensidad')
            plt.legend()

            #Perfil de intensidad columna
            for i in range(len(imagenes)):
                perfilesY.append(imagenes[i][:,y])

            plt.figure(figsize=(10,5))
            colors = ['blue', 'red', 'green']
            labels = ['Original', 'Filtro PasaBajos', 'Filtro Bilateral']
            for i, profile in enumerate(perfilesY):
                plt.plot(perfilesY[i], color=colors[i], label=labels[i])
            plt.title(f'Perfiles de Grises de intesidad de la columna {y}')
            plt.grid(True,'minor')
            plt.xlabel('x')
            plt.ylabel('Intensidad')
            plt.legend()

            plt.show()

    cv.namedWindow('Seleccionar perfiles')
    cv.setMouseCallback('Seleccionar perfiles', on_click)

    cv.imshow('Seleccionar perfiles', imagen1)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

# Llamar a la función para extraer y visualizar los perfiles
mariposa = np.array(mariposa)
mariposa_PB = np.array(mariposa_PB)
mariposa_BI = np.array(mariposa_BI)
# print(mariposa.shape) #601 filas y 799 columnas
# print(mariposa_PB.shape)
# print(mariposa_BI.shape)
extraer_perfiles(mariposa, mariposa_PB, mariposa_BI)

