import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils import ventana_trackbars
# Ejercicio 1: Modelos de color y analisis
# 1. El archivo ‘patron.tif’ corresponde a un patron de colores que varıan por
    # columnas de rojo a azul. En este ejercicio se estudiara la informacion que contienen 
    #las componentes de los diferentes modelos de color:
ruta = "Imagenes_Ej/patron.tif"
imagen = cv.imread(ruta)

imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
hue, saturation, value = cv.split(imagen_hsv)
rojo, verde, azul = cv.split(imagen)

# • Visualice el patron junto a las componentes [R, G, B] y [H, S, V]
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(imagen[:, :, ::-1])
plt.title('Imagen original (RGB)')

plt.subplot(2, 3, 2)
plt.imshow(rojo, cmap='gray')
plt.title('Componente R')

plt.subplot(2, 3, 3)
plt.imshow(verde, cmap='gray')
plt.title('Componente G')

plt.subplot(2, 3, 4)
plt.imshow(azul, cmap='gray')
plt.title('Componente B')

plt.subplot(2, 3, 5)
plt.imshow(hue, cmap='gray')
plt.title('Componente H')

plt.subplot(2, 3, 6)
plt.imshow(saturation, cmap='gray')
plt.title('Componente S')

plt.tight_layout()
plt.show()

# • Analice como varıa la imagen en funcion de los valores de sus planos de
    # color. ¿Que informacion brinda cada canal?
# • Modifique las componentes H, S e V de la imagen para obtener un patron
    # en RGB que cumpla con las siguientes condiciones:
    # – Variacion de matices de azul a rojo.
    # – Saturacion y brillo maximos.

# • Vizualice la nueva imagen y sus componentes en ambos modelos. Analice y
    # saque conclusiones.

ventana = True
if ventana:
    variables_trackbar = ['hue','saturation','value']
    parametros_trackbar = [[0,360],[0,100],[0,255]]

    def transformacion(imagen,valores_trackbar):#Reconstruir la imagen con los valores transformados
        hue = valores_trackbar[0]
        saturation = valores_trackbar[1]
        value = valores_trackbar[2]
        imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
        imagen_hsv = np.array(imagen_hsv)
        imagen_hsv[:, :, 0] += hue
        imagen_hsv[:, :, 1] += saturation
        maxValue = np.max(imagen_hsv[:,:,2])
        imagen_hsv[:,:,2] -= maxValue
        imagen_hsv[:, :, 2] += value
        imagen_rgb = cv.cvtColor(imagen_hsv, cv.COLOR_HSV2BGR)
        return imagen_rgb[:,:,::-1]

    ventana_trackbars(imagen[:,:,::-1], variables_trackbar, parametros_trackbar, transformacion)



# 2. Genere una función cuyo resultado sea una imagen donde los pixeles tengan los
# colores complementarios a los de la original. Utilice las componentes del modelo
# HSV y la imagen ‘rosas.jpg’.
ruta2 = "Imagenes_Ej/rosas.jpg"
imagen2 = cv.imread(ruta2)

imagen2_hsv = cv.cvtColor(imagen2, cv.COLOR_HSV2BGR)

imagen2_hsv_complementario = imagen2_hsv.copy()
hue_complementario = (imagen2_hsv[:,:,0] + 90) % 180 
# hue_complementario = (imagen2_hsv[:,:,0] + 180) % 360 
imagen2_hsv_complementario[:, :, 0] = hue_complementario

imagen2_complementario = cv.cvtColor(imagen2_hsv_complementario, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(imagen2[:,:,::-1])

plt.subplot(1, 2, 2)

plt.imshow(imagen2_complementario[:,:,::-1])

plt.show()
plt.tight_layout()

############### TEORIA #############
# Complementarios de colores primarios
# Ya vimos que las parejas de colores complementarios de primarios surgen de la mezcla de los dos primarios restantes:

# ROJO – VERDE
# AZUL – NARANJA
# AMARILLO – VIOLETA

# Complementarios de colores secundarios
# Son los mismos colores que en la lista anterior, sólo que a la inversa. Sólo lo hago para que lo tengas bien claro

# VERDE – ROJO
# NARANJA – AZUL
# VIOLETA - AMARILLO

# Complementarios de colores terciarios
# Ya aprendimos los complementarios de primarios y secundarios. Ahora veremos los complementarios de colores terciarios. Un terciario surge de la mezcla de un color secundario mas un poco de un primario. A continuación, te dejo las parejas de complementarios formada con colores terciarios:

# AMARILLO VERDOSO – ROJO VIOLETA
# ROJO ANARANJADO – AZUL VERDOSO
# AZUL VIOLACEO – AMARILLO ANARANJADO
######################################

# 3. Mejore la funcion para trazar los perfiles de intensidad que realizo en guıas
    # previas, para que en la misma grafica 
# • se visualicen simultaneamente los perfiles de cada canal: R, G y B.
# • se visualicen los perfiles de los canales H, S y V.
def extraer_perfiles_color(imagen):
    def on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print("X: ", x)
            print("Y: ", y)
            perfilesX_rgb = imagen[x, :, ::-1]
            perfilesX_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)[x, :, :]

            plt.figure(figsize=(12, 6))

            # Perfil de intensidad RGB
            plt.subplot(1, 2, 1)
            plt.plot(perfilesX_rgb[:, 0], color='red', label='R')
            plt.plot(perfilesX_rgb[:, 1], color='green', label='G')
            plt.plot(perfilesX_rgb[:, 2], color='blue', label='B')
            plt.title(f'Perfiles de intensidad RGB de la fila {x}')
            plt.xlabel('y')
            plt.ylabel('Intensidad')
            plt.legend()

            # Perfil de intensidad HSV (matiz, saturación, valor)
            plt.subplot(1, 2, 2)
            plt.plot(perfilesX_hsv[:, 0], color='orange', label='H')
            plt.plot(perfilesX_hsv[:, 1], color='purple', label='S')
            plt.plot(perfilesX_hsv[:, 2], color='brown', label='V')
            plt.title(f'Perfiles de intensidad HSV de la fila {x}')
            plt.xlabel('y')
            plt.ylabel('Intensidad')
            plt.legend()

            plt.tight_layout()
            plt.show()

    cv.namedWindow('Seleccionar perfiles')
    cv.setMouseCallback('Seleccionar perfiles', on_click)

    cv.imshow('Seleccionar perfiles', imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()

extraer_perfiles_color(imagen)