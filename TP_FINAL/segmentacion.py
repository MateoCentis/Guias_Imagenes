import cv2 as cv
import numpy as np
from utils import trackbar_transformacion, leer_imagenes_de_carpeta, escribir_imagenes_carpeta
import matplotlib.pyplot as plt
"""
En este archivo se encuentra el código correspondiente a la lectura de las imágenes originales, la segmentación de las mismas
    y la escritura de las imágenes segmentadas
"""
#---------------------------------------------Funciones-------------------------------------------------
def segmentacion_hsv(imagen, rango_hue, rango_saturation, valor_median_blur):
    imagen_median = cv.medianBlur(imagen, valor_median_blur)
    imagen_hsv = cv.cvtColor(imagen_median, cv.COLOR_BGR2HSV)
    
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
    
    return area_no_segmentada #resultado, segmentacion


def segmentacion_hsv_trackbar(imagen, valores_trackbar):
    rango_hue = [valores_trackbar[0], valores_trackbar[1]]
    rango_saturation = [valores_trackbar[2], valores_trackbar[3]]
    rango_value = [valores_trackbar[4], valores_trackbar[5]]
    valor_median_blur = valores_trackbar[6]
    if valor_median_blur % 2 == 0:
        valor_median_blur += 1
    imagen_median = cv.medianBlur(imagen, valor_median_blur)
    imagen_hsv = cv.cvtColor(imagen_median, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]),
        np.logical_and(rango_value[0] <= v, v <= rango_value[1])
    )
    
    mascara = np.uint8(mascara * 255)  
    
    segmentacion = cv.bitwise_and(imagen, imagen, mask=mascara)
    segmentacion[np.where((segmentacion!=[0,0,0]).all(axis=2))] = [0,0,255]  # Pintamos en rojo
    
    mascara_inversa = cv.bitwise_not(mascara)
    area_no_segmentada = cv.bitwise_and(imagen, imagen, mask=mascara_inversa)
    
    resultado = cv.add(segmentacion, area_no_segmentada)
    
    return mascara_inversa

def transformacion(imagen, valores_trackbar):
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)
    h_promedio = np.mean([np.mean(h[:,0,]), np.mean(h[:,-1]), np.mean(h[0,:]), np.mean(h[-1,:])])
    s_promedio = np.mean([np.mean(s[:,0,]), np.mean(s[:,-1]), np.mean(s[0,:]), np.mean(s[-1,:])])
    color_promedio = [h_promedio, s_promedio]
    print(color_promedio)
    mascara = segmentacion_hsv_trackbar(imagen, valores_trackbar[0:7])
    imagen_procesada = cv.bitwise_and(imagen, imagen, mask=mascara)

    return imagen_procesada

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    imagen = cv.imread("Datos/Potato/Potato___Late_Blight/00b1f292-23dd-44d4-aad3-c1ffb6a6ad5a___RS_LB 4479.JPG")
    probar_segmentaciones = False
    if probar_segmentaciones:
        imagen2 = cv.imread("Datos/Potato/Potato___Early_Blight/7ea05f87-83d3-405e-84ee-f637426f4bab___RS_Early.B 6715.jpg")
        imagen3 = cv.imread("Datos/Potato/Potato___Early_Blight/7ba072ea-9bb3-4b5e-8fff-82758ff3f722___RS_Early.B 8262.jpg")

        variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1','value0','value1','median_blur','R','G','B','radio']
        parametros_trackbar = [[0,360],[0,360],[0,255],[0,255],[0,255],[0,255],[1,31],[0,255],[0,255],[0,255],[0,255]]
        # Obtención de parámetros a través de la prueba de valores con trackbars
        trackbar_transformacion(imagen, 
                        variables_trackbar = variables_trackbar, 
                        parametros_trackbar = parametros_trackbar, 
                        transformacion = transformacion)

    #---------------------------------------------Early-------------------------------------------------
    early = False
    if early:
        print("EARLY")
        # Leemos las imagenes
        carpeta_path_early = "Datos/Potato/Potato___Early_Blight"
        imagenes_early = leer_imagenes_de_carpeta(carpeta_path_early)

        # Segmentamos las imágenes
        rango_hue = [90,198]
        rango_saturation = [0,139]
        imagenes_segmentadas = []
        for i in range(len(imagenes_early)):
            imagen = imagenes_early[i]
            imagen_segmentada = segmentacion_hsv(imagen, rango_hue, rango_saturation,7)
            imagenes_segmentadas.append(imagen_segmentada)

        # Escribimos las imágenes segmentadas
        carpeta_path_early_segmentada = "Datos/Potato/Early_Blight_segmentadas"
        escribir_imagenes_carpeta(imagenes_segmentadas, carpeta_path_early_segmentada)
    #---------------------------------------------Healthy-------------------------------------------------
    healthy = False
    if healthy:
        print("HEALTHY")
        carpeta_path_healthy = "Datos/Potato/Potato___Healthy"
        imagenes_healthy = leer_imagenes_de_carpeta(carpeta_path_healthy)

        rango_hue = [0,181]
        rango_saturation = [0,41]
        imagenes_segmentadas = []
        for i in range(len(imagenes_healthy)):
            imagen = imagenes_healthy[i]
            imagen_segmentada = segmentacion_hsv(imagen, rango_hue, rango_saturation, 11)
            imagenes_segmentadas.append(imagen_segmentada)

        # Escribimos las imágenes segmentadas
        carpeta_path_healthy_segmentada = "Datos/Potato/Healthy_segmentadas"
        escribir_imagenes_carpeta(imagenes_segmentadas, carpeta_path_healthy_segmentada)
    #---------------------------------------------Late-------------------------------------------------
    late = False
    if late:
        print("LATE")
        carpeta_path_late = "Datos/Potato/Potato___Late_Blight"
        imagenes_late = leer_imagenes_de_carpeta(carpeta_path_late)

        rango_hue = [0,181]
        rango_saturation = [0,41]
        imagenes_segmentadas = []
        for i in range(len(imagenes_late)):
            imagen = imagenes_late[i]
            imagen_segmentada = segmentacion_hsv(imagen, rango_hue, rango_saturation, 11)

            imagenes_segmentadas.append(imagen_segmentada)

        # Escribimos las imágenes segmentadas
        carpeta_path_Late_segmentadas = "Datos/Potato/Late_segmentadas_otra_forma"
        escribir_imagenes_carpeta(imagenes_segmentadas, carpeta_path_Late_segmentadas)