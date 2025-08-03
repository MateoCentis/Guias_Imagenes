import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import ventana_trackbars
# Función que genera la segmentación en HSV de la imagen
def segmentacion_hsv(imagen, rango_hue, rango_saturation):

    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    
    h, s, _ = cv.split(imagen_hsv)
    
    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1])
    )
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    segmentacion = cv.bitwise_and(imagen, imagen, mask=mascara)
    segmentacion[np.where((segmentacion!=[0,0,0]).all(axis=2))] = [0,0,255]  # Pintamos en rojo
    
    mascara_inversa = cv.bitwise_not(mascara)
    area_no_segmentada = cv.bitwise_and(imagen, imagen, mask=mascara_inversa)
    
    resultado = cv.add(segmentacion, area_no_segmentada)
    
    return resultado, segmentacion


def segmentacion_hsv_trackbar(imagen, valores_trackbar):
    rango_hue = [valores_trackbar[0], valores_trackbar[1]]
    rango_saturation = [valores_trackbar[2], valores_trackbar[3]]
    rango_value = [valores_trackbar[4], valores_trackbar[5]]
    
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]),
        np.logical_and(rango_value[0] <= v, v <= rango_value[1])
    )
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    # Pintamos el área segmentada en rojo
    segmentacion = cv.bitwise_and(imagen, imagen, mask=mascara)
    segmentacion[np.where((segmentacion!=[0,0,0]).all(axis=2))] = [0,0,255]  # Pintamos en rojo
    
    # Convertimos la máscara a un formato donde el área segmentada sea blanca y el resto negro
    mascara_inversa = cv.bitwise_not(mascara)
    area_no_segmentada = cv.bitwise_and(imagen, imagen, mask=mascara_inversa)
    
    # Combinamos el área segmentada en rojo con el resto de la imagen
    resultado = cv.add(segmentacion, area_no_segmentada)
    
    return resultado

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

ruta = "Imagenes_Ej/Deforestacion.png"
imagen = cv.imread(ruta)

x0 = 153
y0 = 274
x1 = 728
y1 = 703
imagen_delimitada = imagen[y0:y1,x0:x1].copy()

plt.imshow(imagen)
plt.show()

# Pre-procesamiento para hacer más facil la segmentacion
sizeMascara = 18
mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
imagen_delimitada_preprocesada = cv.filter2D(imagen_delimitada, -1, mascara_promedio)

mostrar_trackbar = True
if mostrar_trackbar:
    # Trackbar
    variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1','value0','value1']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255],[0,255],[0,255]]

    # Obtención de parámetros a través de la prueba de valores con trackbars
    ventana_trackbars(imagen_delimitada_preprocesada, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_hsv_trackbar)

################################################################################################
#Segmente y resalte en algún tono de rojo el área deforestada 
################################################################################################
rango_hue = [0, 67] #Valores obtenidos a través de vista de trackbars
rango_saturation = [0, 55]
imagen_delimitada_segmentada, segmentacion = segmentacion_hsv(imagen_delimitada_preprocesada, rango_hue, rango_saturation)

segmentacion = cv.cvtColor(segmentacion,cv.COLOR_BGR2RGB)
imagen_delimitada_final = cv.add(imagen_delimitada, segmentacion)
plt.imshow(imagen_delimitada_final)

plt.show()

################################################################################################################################
#Calcule el área total (hectáreas) de la zona delimitada, el área de la zona que tiene monte y el área de la zona deforestada.
################################################################################################################################


def calcular_area(imagen_total, imagen_segmentada, unidad_imagen_a_metros):
    # Área total
    ancho_imagen = imagen_total.shape[1] 
    alto_imagen = imagen_total.shape[0]   
    area_total_pixeles = ancho_imagen * alto_imagen  
    area_total_metros_cuadrados = area_total_pixeles * (unidad_imagen_a_metros ** 2)  
    
    # Área segmentada
    area_segmentada_pixeles = np.sum(imagen_segmentada != 0)  
    area_segmentada_metros_cuadrados = area_segmentada_pixeles * (unidad_imagen_a_metros ** 2)  
    
    # Área no segmentada (restamos)
    area_no_segmentada_metros_cuadrados = area_total_metros_cuadrados - area_segmentada_metros_cuadrados
    
    return area_segmentada_metros_cuadrados, area_no_segmentada_metros_cuadrados, area_total_metros_cuadrados

unidad_imagen_a_metros = 200/100 # #Usando la escala como referencia (100 unidades de la imagen = 200 m => 1 unidad -> 2 metros)

area_deforestada, area_monte, area_total = calcular_area(imagen_delimitada, segmentacion, unidad_imagen_a_metros)

print("Area deforestada", area_deforestada) #169332

print("Area monte", area_monte) #817368

print("Area total", area_total) #986700
