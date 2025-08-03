import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 1. Manejo de histograma: la imagen ‘chairs oscura.jpg’ posee poca luminosidad. Usted debe mejorar la imagen a partir de la 
    #ecualizacion de histograma,comparando los efectos de realizarla en RGB (por planos), 
        #en HSV (canal V) y en HSI (canal I).
ruta = "Imagenes_Ej/chairs.jpg"
ruta_oscura = "Imagenes_Ej/chairs_oscura.jpg"
imagen = cv.imread(ruta)
imagen_oscura = cv.imread(ruta_oscura)

# imagen_oscura_hsv = cv.cvtColor(imagen_oscura,cv.COLOR_BGR2HSV)
# imagen_oscura_hsv_eq = cv.equalizeHist(imagen_oscura_hsv)

# imagen_oscura_hsi = imagen_oscura_hsv.copy()
# imagen_oscura_hsi[:,:,2] = (imagen_oscura[:,:,0]+imagen_oscura[:,:,1]+imagen_oscura[:,:,2])/3
# imagen_oscura_hsi_eq = cv.equalizeHist(imagen_oscura_hsi)
# imagen_oscura_hsi_eq = cv.cvtColor(imagen_oscura_hsi_eq,cv.color_hs)
# imagen_oscura_rgb = cv.cvtColor(imagen_oscura,cv.COLOR_BGR2RGB)
# imagen_oscura_rgb_eq = cv.equalizeHist(imagen_oscura_rgb)

# Función para ecualización de histograma en diferentes espacios de color
def ecualizar_histograma(imagen, espacio_de_color):
    if espacio_de_color == 'RGB':
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        for i in range(3):
            imagen_filtrada[:,:,i] = cv.equalizeHist(imagen_filtrada[:,:,i])
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_RGB2BGR)
    elif espacio_de_color == 'HSV':
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
        imagen_filtrada[:,:,2] = cv.equalizeHist(imagen_filtrada[:,:,2])
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_HSV2BGR)
    elif espacio_de_color == 'HSI': # Arreglar esto 
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
        intensidad = np.sum(imagen,axis=2)/3
        intensidad_filtrada = cv.equalizeHist(intensidad.astype(np.uint8))
        imagen_filtrada[:,:,2] = intensidad_filtrada
        #reemplazo value por intensidad y ecualizo intensidad
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_HSV2BGR)
    return imagen_filtrada

img_rgb_eq = ecualizar_histograma(imagen_oscura, 'RGB')
img_hsv_eq = ecualizar_histograma(imagen_oscura, 'HSV')
img_hsi_eq = ecualizar_histograma(imagen_oscura, 'HSI')

# • Visualice la imagen original ‘chairs.jpg’, comparela con las imagenes realzadas y discuta los resultados.
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(cv.cvtColor(imagen, cv.COLOR_BGR2RGB))
axes[0, 0].set_title('Imagen Original')

axes[0, 1].imshow(cv.cvtColor(img_rgb_eq, cv.COLOR_BGR2RGB))
axes[0, 1].set_title('Ecualización en RGB')

axes[1, 0].imshow(cv.cvtColor(img_hsv_eq, cv.COLOR_BGR2RGB))
axes[1, 0].set_title('Ecualización en HSV')

axes[1, 1].imshow(cv.cvtColor(img_hsi_eq, cv.COLOR_BGR2RGB))
axes[1, 1].set_title('Ecualización en HSI')

plt.show()

# • Repita el proceso para otras imagenes de bajo contraste (por ejemplo ‘flowers_oscura.tif’) y analice los resultados.
ruta_flores = "Imagenes_Ej/flowers_oscura.tif"
flores_oscura = cv.imread(ruta_flores)

img_rgb_eq = ecualizar_histograma(flores_oscura, 'RGB')
img_hsv_eq = ecualizar_histograma(flores_oscura, 'HSV')
img_hsi_eq = ecualizar_histograma(flores_oscura, 'HSI')

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(cv.cvtColor(flores_oscura, cv.COLOR_BGR2RGB))
axes[0, 0].set_title('Imagen Original')

axes[0, 1].imshow(cv.cvtColor(img_rgb_eq, cv.COLOR_BGR2RGB))
axes[0, 1].set_title('Ecualización en RGB')

axes[1, 0].imshow(cv.cvtColor(img_hsv_eq, cv.COLOR_BGR2RGB))
axes[1, 0].set_title('Ecualización en HSV')

axes[1, 1].imshow(cv.cvtColor(img_hsi_eq, cv.COLOR_BGR2RGB))
axes[1, 1].set_title('Ecualización en HSI')

plt.show()

# 2. Realce mediante acentuado: utilice la imagen ‘camino.tif’ que se observa desenfocada. Usted debe mejorar la imagen 
    #aplicando un filtro pasa altos de suma
# 1. Compare los resultados de procesar la imagen en los modelos RGB, HSV y HSI
ruta_camino = "Imagenes_Ej/camino.tif"
imagen_camino = cv.imread(ruta_camino)


def filtro_pasa_alto(imagen, espacio_de_color):
    mascara_suma = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    if espacio_de_color == 'RGB':
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        for i in range (3):
            imagen_filtrada[:,:,i] = cv.filter2D(imagen_filtrada[:,:,i],-1,mascara_suma)
        
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_RGB2BGR)

    elif espacio_de_color == 'HSV':
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
       
        imagen_filtrada[:,:,2] = cv.filter2D(imagen_filtrada[:,:,2],-1,mascara_suma)
       
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_HSV2BGR)

    elif espacio_de_color == 'HSI': 
        imagen_filtrada = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
        
        intensidad = np.sum(imagen,axis=2)/3
        imagen_filtrada[:,:,2] = intensidad.astype(np.uint8)
        
        imagen_filtrada[:,:,2] = cv.filter2D(imagen_filtrada[:,:,2],-1,mascara_suma)
        
        imagen_filtrada = cv.cvtColor(imagen_filtrada, cv.COLOR_HSV2BGR)

    return imagen_filtrada


img_rgb_eq = filtro_pasa_alto(imagen_camino, 'RGB')
img_hsv_eq = filtro_pasa_alto(imagen_camino, 'HSV')
img_hsi_eq = filtro_pasa_alto(imagen_camino, 'HSI')

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(cv.cvtColor(imagen_camino, cv.COLOR_BGR2RGB))
axes[0, 0].set_title('Imagen Original')

axes[0, 1].imshow(cv.cvtColor(img_rgb_eq, cv.COLOR_BGR2RGB))
axes[0, 1].set_title('Filtrado en RGB')

axes[1, 0].imshow(cv.cvtColor(img_hsv_eq, cv.COLOR_BGR2RGB))
axes[1, 0].set_title('Filtrado en HSV')

axes[1, 1].imshow(cv.cvtColor(img_hsi_eq, cv.COLOR_BGR2RGB))
axes[1, 1].set_title('Filtrado en HSI')

plt.show()