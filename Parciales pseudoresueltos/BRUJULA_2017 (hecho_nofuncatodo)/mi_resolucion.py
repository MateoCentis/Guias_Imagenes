import cv2 as cv
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from utils import segmentacion_hsv, trackbar_transformacion
import math
#---------------------------------------------Funciones-------------------------------------------------
def segmentacion_RGB(imagen, color, radio, color_dentro_radio = None, color_fuera_radio = None):
    imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    
    centroide = np.array(color)
    distancias = np.linalg.norm(imagen_rgb - centroide, axis=-1)

    mascara_rgb = np.where(distancias <= radio, 1, 0).astype(np.uint8)

    if color_dentro_radio is None or color_fuera_radio is None: 
        color_dentro_radio = (255,0,0) #Rojo
        color_fuera_radio = (0,255,0) #Verde
    
    imagen_rgb_segmentada = np.where(mascara_rgb[...,None], color_dentro_radio, color_fuera_radio)
    
    mascara_rgb = np.where(distancias <= radio, 255, 0).astype(np.uint8) #Donde las distancias sean <= radio 1 sino 0)
    
    imagen_rgb_segmentada = cv.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara_rgb)
    
    return imagen_rgb_segmentada, mascara_rgb

def segmentacion_RGB_trackbar(imagen,valores_trackbar):
    R = valores_trackbar[0]
    G = valores_trackbar[1]
    B = valores_trackbar[2]
    color = (R,G,B)
    radio = valores_trackbar[3]

    imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    centroide = np.array(color)
    distancias = np.linalg.norm(imagen_rgb - centroide, axis=-1)

    mascara_rgb = np.where(distancias <= radio, 1, 0).astype(np.uint8)

    # if color_dentro_radio is None or color_fuera_radio is None: 
    color_dentro_radio = (255,0,0) #Rojo
    color_fuera_radio = (0,255,0) #Verde
    
    imagen_rgb_segmentada = np.where(mascara_rgb[...,None], color_dentro_radio, color_fuera_radio)
    
    mascara_rgb = np.where(distancias <= radio, 255, 0).astype(np.uint8) #Donde las distancias sean <= radio 1 sino 0)
    
    imagen_rgb_segmentada = cv.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara_rgb)
    
    return imagen_rgb_segmentada#, mascara_rgb

#---------------------------------------------Comienzo-------------------------------------------------

#Lectura de imágenes
imagen1 = cv.imread("BRUJULA_2017 (hecho_nofuncatodo)/1.png") 
imagen2 = cv.imread("BRUJULA_2017 (hecho_nofuncatodo)/2.png") 
imagen3 = cv.imread("BRUJULA_2017 (hecho_nofuncatodo)/3.png") 
imagen4 = cv.imread("BRUJULA_2017 (hecho_nofuncatodo)/4.png")

imagenes = [imagen1, imagen2, imagen3, imagen4]

######################################IDEA##################################################
#Se quiere encontrar un ángulo
    #Para este ángulo se necesita entre N (norte)
    #Y donde está apuntándo la brújula (aguja roja)
    #Para encontrar el ángulo se necesita:
        #1. Encontrar el centro de la imagen
        #2. Encontrar el centro de la aguja roja
            #Con 1. y 2. hacemos dos vectores:
                # Vector aguja
                # Vector Norte
        #3. Calcular el ángulo entre el centro de la imagen y el centro de la aguja roja 
###########################################################################################
# 1. Centro de la imagen (se hardcodea ya que todas son del mismo tamaño)
    #Esto hace el método vulnerable a imagenes de diferentes tamaños
centro_x = 200
centro_y = 200
P0 = (centro_x, centro_y)

# 2. Segmentación en color - se hace con trackbars para obtener los mejores parametros
ver_trackbars = False
if ver_trackbars:
    variables_trackbar = ['R', 'G', 'B', 'radio']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255]]

    trackbar_transformacion(imagen3, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_RGB_trackbar)  
#Podemos segmentar por color, luego a esto aplicar hit-or-miss para obtener solo lo que nosotros queremos ver
color = (220,0,0)
radio = 120
#Acá se obtienen las máscaras con toda la información que necesitamos, ahora hay que procesarla
mascaras = []
for i in range(len(imagenes)):
    imagen = imagenes[i]
    _, mascara = segmentacion_RGB(imagen,color, radio, None, None)
    mascaras.append(mascara)


#Definición de kernels para obtener características de la imagen
solo_norte = []
solo_aguja = []
for i in range(len(mascaras)):
    mascara = mascaras[i]
    kernel = np.zeros((3,3),dtype=np.uint8)
    # Definición de kernels para obtener imágenes solo con la N y solo con aguja
    if i < 1: 
        kernel = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], np.uint8)
        kernel_aguja = np.array([
            [0,0,1],
            [1,1,1],
            [1,0,0]
        ])
    elif i < 2: 
        kernel = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], np.uint8)
        kernel_aguja = np.array([
            [1,1,0],
            [0,1,0],
            [0,1,1]
        ])
    elif i < 3:
        kernel = np.array([
            [0,0,1],
            [0,1,0],
            [1,0,0]
        ], np.uint8)
        kernel_aguja = np.array([
            [1,1,0],
            [0,1,0],
            [0,1,1]
        ])
    else:
        kernel = np.array([
            [0,1,0],
            [0,1,0],
            [0,1,0]
        ], np.uint8)
        kernel_aguja = np.array([
            [0,1,1],
            [0,1,0],
            [1,1,0]
        ])

    resultado = cv.morphologyEx(mascara,cv.MORPH_HITMISS,kernel,iterations=10)
    semilla = resultado.copy()
    reconstruido = morphology.reconstruction(semilla,mascara,method='dilation')
    solo_norte.append(reconstruido)
    
    morfologia_aguja = cv.morphologyEx(mascara,cv.MORPH_HITMISS, kernel_aguja, iterations=5)
    reconstruido_aguja = morphology.reconstruction(morfologia_aguja,mascara, method='dilation')
    # plt.imshow(reconstruido_aguja,cmap='gray')
    # plt.show()
    solo_aguja.append(reconstruido_aguja)

#Una vez tenemos solo agujas y solo norte encontramos los puntos en espacio y armamos los vectores        
puntos_norte = []
for i in range(len(solo_norte)):
    mascara = solo_norte[i].astype(np.uint8)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mascara)
    for j, (x, y, w, h, area) in enumerate(stats):
        if j == 0:
            continue
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        puntos_norte.append((center_x, center_y))

puntos_aguja = []

for i in range(len(solo_aguja)):
    mascara = solo_aguja[i].astype(np.uint8)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mascara)
    for j, (x, y, w, h, area) in enumerate(stats):
        if j == 0:
            continue
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        print("Brujula ", str(i+1), " - componete ",str(i+1),": ", center_x, center_y)
        puntos_aguja.append((center_x, center_y))

#Con cada punto norte, punto aguja y punto central hacemos vectores y medimos los ángulos
for i in range(len(puntos_norte)):
    aguja = puntos_aguja[i]
    norte = puntos_norte[i]

    vector_norte = [norte[1] - P0[1], norte[0] - P0[0]]

    vector_aguja = [aguja[1] - P0[1], aguja[0] - P0[0]]
    producto_punto_unitario = np.dot(vector_aguja,vector_norte)/(np.linalg.norm(vector_aguja)*np.linalg.norm(vector_norte))
    angulo = math.acos(producto_punto_unitario)*(180/math.pi)
    # angulo = np.arctan2(vector_norte[1], vector_norte[0]) - np.arctan2(vector_aguja[1], vector_aguja[0])
    print("Ángulo Brújula ",str(i+1),": ", 360-angulo)


    #1: 260°
    #2: 80°
    #3: 170°
    #4: 80°