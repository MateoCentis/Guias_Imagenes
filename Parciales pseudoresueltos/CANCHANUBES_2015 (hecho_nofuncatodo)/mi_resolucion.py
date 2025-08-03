
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from utils import trackbar_transformacion
# sys.path.append('../../') #para la búsqueda de módulos

#---------------------------------------------Funciones-------------------------------------------------
def leer_archivos_directorio(directorio):
    # Verifica si el directorio existe
    if not os.path.isdir(directorio):
        raise ValueError(f"El directorio '{directorio}' no existe")
    
    # Lista para almacenar los nombres de los archivos
    nombres_archivos = []

    # Recorre el directorio y agrega los nombres de los archivos a la lista
    for nombre in os.listdir(directorio):
        ruta_completa = os.path.join(directorio, nombre)
        if os.path.isfile(ruta_completa):
            nombres_archivos.append(nombre)
    
    return nombres_archivos

def leer_imagenes_de_carpeta(carpeta):
    imagenes = []
    for nombre_archivo in os.listdir(carpeta):
        path = os.path.join(carpeta, nombre_archivo)
        imagen = cv.imread(path)
        if imagen is not None:
            imagenes.append(imagen)
    return imagenes
# Función para escribir las imagenes en una carpeta
def escribir_imagenes_carpeta(imagenes, carpeta):
  if not os.path.exists(carpeta):
    os.makedirs(carpeta)

  for i, image in enumerate(imagenes):
    nombre_archivo = os.path.join(carpeta, f"imagen_segmentada_{i+1}.jpg") 
    cv.imwrite(nombre_archivo, image)

def aplicar_hough(imagen,umbrales_canny : tuple[int,int], rho=1, theta=np.pi/180, umbral_hough=200):
    imagen_resultado = imagen.copy()
    umbral0 = umbrales_canny[0]
    umbral1 = umbrales_canny[1]
    bordes = cv.Canny(imagen, umbral0, umbral1)
    lineas = cv.HoughLines(bordes, rho, theta, umbral_hough)
    
    angulos_por_linea = []
    for linea in lineas:
        rho, theta = linea[0]
        angulos_por_linea.append(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(imagen_resultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return imagen_resultado, lineas, angulos_por_linea

def calcular_angulos_rotacion(lineas):
    angulos = []
    for line in lineas:
        for x1, y1, x2, y2 in line:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angulos.append(angle)
    
    return angulos
def aplicar_hough_probabilistico(imagen, umbrales_canny : tuple[int,int], rho=1, theta=np.pi/180,umbral_hough=200, minLineLength=100, maxLineGap=10):
    imagen_resultado = imagen.copy()
    umbral0 = umbrales_canny[0]
    umbral1 = umbrales_canny[1]
    bordes = cv.Canny(imagen, umbral0, umbral1)
    lineas = cv.HoughLinesP(bordes, rho, theta, umbral_hough, minLineLength, maxLineGap)
    angulos = calcular_angulos_rotacion(lineas)
    return lineas, angulos

def rotate_image(image, angle):
    # Get the image size
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

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
    # I = (imagen[:,:,0] + imagen[:,:,1] + imagen[:,:,2])/3

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
    
    return segmentacion


def rotate(img, angle):
    """Rotación de la imagen sobre el centro"""
    r = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1.0)
    # The corrected line is below
    return cv.warpAffine(img, r, (img.shape[1], img.shape[0]))

def detectar_rotacion(imagen, umbral1=50, umbral2=150): #esto anda pero ni idea
    edges = cv.Canny(imagen, umbral1, umbral2)
    # Detecta las líneas con la transformada de Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    mean_angle = np.mean(angles)
    return mean_angle
#---------------------------------------------Resolución-------------------------------------------------

directorio = "Parciales/CANCHANUBES_2015/imgs"

nombres_archivos = leer_archivos_directorio(directorio)
imagenes = leer_imagenes_de_carpeta(directorio)

#obtengo una imagen promedio de todas las imagenes para limpiar imagen
imagen_promedio = np.mean(imagenes, axis=0).astype(np.uint8)
# plt.imshow(imagen_promedio)
# plt.show()
#lo que se podría hacer también es umbralizar las imagenes para sacar las nubes y luego hacer promedio sin nubes?
imagen_promedio_gris = cv.cvtColor(imagen_promedio, cv.COLOR_BGR2GRAY)
# cv.imshow("Imagen promediada",imagen_promedio)  #MOSTRAR IMAGEN LIMPIA

#Para obtener la línea de las gradas techadas podemos segmentar para obtener solo esa parte ( NO FUNCÓ )
ver_trackbars = False
if ver_trackbars:
    variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1','value0','value1','median']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255],[0,255],[0,255],[1,31]]

    # # Obtención de parámetros a través de la prueba de valores con trackbars
    trackbar_transformacion(imagen_promedio, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_hsv_trackbar) 

# ¿Cómo solo agarrar la parte que quiero?
angulo_rotacion = detectar_rotacion(imagen_promedio_gris)
imagen_rotada = rotate(imagen_promedio, angulo_rotacion)
cv.imshow("ROTADA", imagen_rotada)
cv.waitKey(0)
# bordes = cv.Canny(imagen_promedio_gris, 140, 255)
# cv.imshow("Bordes", bordes)

# umbrales_canny = [140,255]
# # lineas, angulos = aplicar_hough_probabilistico(imagen_promedio_gris, umbrales_canny)
# imagen_hough, _, angulos = aplicar_hough(imagen_promedio_gris, umbrales_canny,umbral_hough=150)
# cv.imshow("HOUGH", imagen_hough)
# print("------")
# print(angulos)
# imagen_rotada = rotate_image(imagen_promedio,angulos[0])
# cv.imshow("IMAGEN ROTADA", imagen_rotada)



cv.waitKey(0)

