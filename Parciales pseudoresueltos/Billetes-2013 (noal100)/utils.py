import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.fft import fftshift, fft2
import os
import math
from skimage import morphology
import imutils

def calcular_area_imagen_binaria(imagen_binaria):
    if len(np.unique(imagen_binaria)) > 2:
        raise ValueError("La imagen no es binaria, convertila")
    
    area = np.sum(imagen_binaria == 255)

    return area
    
def segmentacion_hsv_trackbar_mascara_inversa(imagen_BGR, valores_trackbar):
    rango_hue = [valores_trackbar[0], valores_trackbar[1]]
    rango_saturation = [valores_trackbar[2], valores_trackbar[3]]
    imagen_hsv = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]))
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    mascara_inversa = cv.bitwise_not(mascara)
    area_no_segmentada = cv.bitwise_and(imagen_BGR, imagen_BGR, mask=mascara_inversa)
    
    return area_no_segmentada

def segmentacion_hsv_trackbar(imagen_BGR, valores_trackbar):
    rango_hue = [valores_trackbar[0], valores_trackbar[1]]
    rango_saturation = [valores_trackbar[2], valores_trackbar[3]]
    imagen_hsv = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]))
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    segmentacion = cv.bitwise_and(imagen_BGR, imagen_BGR, mask=mascara)
    
    return segmentacion


########################################################################################
#                                  Funciones para parcial
########################################################################################
def segmentar_RGB(imagen_BGR,R,G,B,radio):
    color = (R,G,B)

    imagen_rgb = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2RGB)
    centroide = np.array(color)
    distancias = np.linalg.norm(imagen_rgb - centroide, axis=-1)

    mascara_rgb = np.where(distancias <= radio, 1, 0).astype(np.uint8)
    

    imagen_bgr_segmentada = cv.bitwise_and(imagen_BGR, imagen_BGR, mask=mascara_rgb)    
    
    return imagen_bgr_segmentada, mascara_rgb

def segmentacion_hsv(imagen_BGR, rango_hue, rango_saturation):
    imagen_hsv = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2HSV)
    h, s, _ = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]))
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    segmentacion = cv.bitwise_and(imagen_BGR, imagen_BGR, mask=mascara)
    
    return segmentacion, mascara

def trackbar_hough_lineasP(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        tresh1_canny = valores_trackbar[0]
        tresh2_canny = valores_trackbar[1]
        treshold = valores_trackbar[2]
        minima_longitud_linea = valores_trackbar[3]
        maximo_gap = valores_trackbar[4]
        bordes = cv.Canny(imagen, tresh1_canny, tresh2_canny)
        lineas = cv.HoughLinesP(bordes, 1, np.pi/180, treshold, minLineLength=minima_longitud_linea, maxLineGap=maximo_gap)
        if lineas is not None:
                for line in lineas:
                    x1, y1, x2, y2 = line[0]
                    cv.line(imagen_salida, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        return imagen_salida
    variables_trackbar = ['tresh1_canny','tresh2_canny','treshold','minima_longitud_linea','maximo_gap']
    parametros_trackbar = [[0,255],[0,255],[0,255],[0,255],[0,255]]
    trackbar_transformacion(imagen, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = transformacion)
    
def divisionesH(imagen, margen):
    """
    Divide la imagen en secciones horizontales basadas en la intensidad de los píxeles y un umbral ajustado por margen.

    Parámetros:
    imagen (numpy.ndarray): Matriz 2D que representa la imagen en escala de grises.
    margen (float): Valor porcentual que ajusta el umbral de división.

    Retorno:
    list: Lista de listas, donde cada sublista contiene dos valores [inicio, fin] que representan
          los índices de las filas que conforman cada sección.
    """
    H, W = imagen.shape

    suma = []
    min_val = 99999
    max_val = 0

    # Calcula la suma de los valores de píxeles para cada fila
    for y in range(H):
        suma.append(0)
        for x in range(W):
            suma[y] += imagen[y, x]
        suma[y] = suma[y] / W
        if suma[y] > max_val:
            max_val = suma[y]
        if suma[y] < min_val:
            min_val = suma[y]

    # Ajusta el valor mínimo basado en el margen
    min_val = min_val + ((max_val - min_val) * margen / 100)

    # Verifica si el primer valor de la suma está por encima del mínimo ajustado
    band = suma[0] > min_val

    ini = 0
    divisiones = []

    # Identifica las transiciones para definir las secciones
    for i in range(len(suma)):
        if suma[i] < min_val and band:
            divisiones.append([ini, i])
            band = False
        elif suma[i] > min_val and not band:
            ini = i
            band = True

    if band:
        divisiones.append([ini, len(suma)])

    return divisiones

def get_border_type(tipo_borde):
    switcher = {
        0: cv.BORDER_CONSTANT,# Pads the image with a constant value (specified by borderValue).
        1: cv.BORDER_REPLICATE,#Repeats the border pixels.
        2: cv.BORDER_REFLECT,# Reflects the border pixels. For example, fedcba|abcdefgh|hgfedcb.
        3: cv.BORDER_WRAP,#Wraps around the image. For example, cdefgh|abcdefgh|abcdefg.
        4: cv.BORDER_REFLECT_101,  # Also cv.BORDER_DEFAULT,  Reflects the border pixels but the border pixel itself is not reflected. For example, gfedcb|abcdefgh|gfedcba.
        5: cv.BORDER_TRANSPARENT,#The pixels beyond the image are not modified.
        6: cv.BORDER_ISOLATED,#Treats all border pixels as isolated pixels. It has no padding and hence is used when no border is needed.
    }
    return switcher.get(tipo_borde, cv.BORDER_CONSTANT)

def trackbar_erosion(imagen,kernel):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        iteraciones = valores_trackbar[0]
        tipo_borde = valores_trackbar[1]
        clave_tipo = get_border_type(tipo_borde)
        valor_borde = valores_trackbar[2]
        imagen_salida = cv.erode(imagen_salida, kernel, iterations=iteraciones, 
                                 borderType=clave_tipo, borderValue=valor_borde)
        return imagen_salida
    
    variables_trackbar = ['iteraciones','tipo_borde','valor_borde']
    parametros_trackbar = [[1,50],[0,6],[0,1]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)
    
def trackbar_hough_lineas_and_canny(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        tresh_canny = valores_trackbar[0]
        tresh_hough = valores_trackbar[1]
        min_theta = valores_trackbar[2]
        max_theta = valores_trackbar[3]

        if min_theta > max_theta:
            min_theta, max_theta = max_theta, min_theta

        bordes = cv.Canny(imagen, tresh_canny, 255)
        
        lines = cv.HoughLines(bordes, 1, np.pi / 180, tresh_hough, min_theta=min_theta, max_theta=max_theta)
        
        line_img = np.zeros_like(bordes)
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(line_img, (x1, y1), (x2, y2), 255, 2)  # Draw lines in white color
        
        line_img = cv.resize(line_img, (imagen.shape[1], imagen.shape[0]))
        combined = cv.bitwise_and(line_img, bordes)

        imagen_salida[combined > 0] = [0, 0, 255]

        return imagen_salida

    variables_trackbar = ['tresh_canny', 'acumulador_hough','min_theta','max_theta']
    parametros_trackbar = [[0, 255],[0, 255], [0,360],[0, 360]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)
    
def trackbar_dilatacion(imagen,kernel):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        iteraciones = valores_trackbar[0]
        tipo_borde = valores_trackbar[1]
        clave_tipo = get_border_type(tipo_borde)
        valor_borde = valores_trackbar[2]
        imagen_salida = cv.dilate(imagen_salida, kernel, iterations=iteraciones, 
                                 borderType=clave_tipo, borderValue=valor_borde)
        return imagen_salida
    
    variables_trackbar = ['iteraciones','tipo_borde','valor_borde']
    parametros_trackbar = [[1,50],[0,6],[0,1]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)
    
def trackbar_hough_lineas(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        tresh_canny = valores_trackbar[0]
        tresh_hough = valores_trackbar[1]
        min_theta = valores_trackbar[2]
        max_theta = valores_trackbar[3]
        if min_theta > max_theta:
            min_theta, max_theta = max_theta, min_theta
        bordes = cv.Canny(imagen, tresh_canny, 255)
        lineas = cv.HoughLines(bordes, 1, np.pi/180, tresh_hough, min_theta=min_theta, max_theta=max_theta)
        
        if lineas is not None:
            for line in lineas:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(imagen_salida, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return imagen_salida

    variables_trackbar = ['tresh_canny', 'acumulador_hough','min_theta','max_theta']
    parametros_trackbar = [[0, 255],[0, 255], [0,360],[0, 360]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)

def trackbar_transformacion_val(imagen, variables_trackbar, parametros_trackbar, transformacion):
    def on_trackbar_change(val=None):
        valores_trackbar = [cv.getTrackbarPos(var, 'Trackbars') for var in variables_trackbar]
        imagen_transformada = transformacion(imagen, valores_trackbar)
        cv.imshow('Transformacion', imagen_transformada)

    cv.namedWindow('Trackbars')
    for var, (min_val, max_val) in zip(variables_trackbar, parametros_trackbar):
        cv.createTrackbar(var, 'Trackbars', min_val, max_val, on_trackbar_change)

    on_trackbar_change()
    cv.waitKey(0)
    cv.destroyAllWindows()
    

def divisionesV(imagen, margen):
    """
    Divide la imagen en secciones verticales basadas en la intensidad de los píxeles y un umbral ajustado por margen.

    Parámetros:
    imagen (numpy.ndarray): Matriz 2D que representa la imagen en escala de grises.
    margen (float): Valor porcentual que ajusta el umbral de división.

    Retorno:
    list: Lista de listas, donde cada sublista contiene dos valores [inicio, fin] que representan
          los índices de las columnas que conforman cada sección.
    """
    H, W = imagen.shape

    suma = []
    min_val = 99999
    max_val = 0

    # Calcula la suma de los valores de píxeles para cada columna
    for x in range(W):
        suma.append(0)
        for y in range(H):
            suma[x] += imagen[y, x]
        suma[x] = suma[x] / H
        if suma[x] > max_val:
            max_val = suma[x]
        if suma[x] < min_val:
            min_val = suma[x]

    # Ajusta el valor mínimo basado en el margen
    min_val = min_val + ((max_val - min_val) * margen / 100)

    # Verifica si el primer valor de la suma está por encima del mínimo ajustado
    band = suma[0] > min_val

    ini = 0
    divisiones = []

    # Identifica las transiciones para definir las secciones
    for i in range(len(suma)):
        if suma[i] < min_val and band:
            divisiones.append([ini, i])
            band = False
        elif suma[i] > min_val and not band:
            ini = i
            band = True

    if band:
        divisiones.append([ini, len(suma)])

    return divisiones


def color_promedio(imagen):
    average_color = np.average(imagen, axis=(0, 1))

    # Convertir el color promedio de BGR a formato HSV
    average_color_bgr = np.uint8([[average_color]])
    average_color_hsv = cv.cvtColor(average_color_bgr, cv.COLOR_BGR2HSV)[0][0]

    return np.uint8(average_color), average_color_hsv #BGR y HSV

def rectangulos_trackbar(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagen_gris = cv.GaussianBlur(imagen_gris, (5,5),0)
        imagen_salida = imagen.copy()
        block_size = valores_trackbar[0]
        if block_size % 2 == 0:
            block_size = block_size + 1
        C = valores_trackbar[1]
        area_min = valores_trackbar[2]
        area_max = valores_trackbar[3]
        metodo_umbralizado = valores_trackbar[4]
        umbral = valores_trackbar[5]
        tresh1_canny = valores_trackbar[6]
        tresh2_canny = valores_trackbar[7]
        if metodo_umbralizado == 0:
            imagen_bin = cv.adaptiveThreshold(imagen_gris, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)
        elif metodo_umbralizado == 1:
            imagen_bin = cv.threshold(imagen_gris, umbral, 255, cv.THRESH_BINARY)[1]
        else:
            imagen_bin = cv.Canny(imagen_gris, tresh1_canny, tresh2_canny)
        
        contornos,_ = cv.findContours(imagen_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            x, y, w, h = cv.boundingRect(contorno)
            area = w*h
            if area_min <= area <= area_max:
                cv.rectangle(imagen_salida, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return imagen_salida
    
    variables_trackbar = ['block_size_metodo0','C_metodo0','area_min','area_max','metodo_umbralizado','umbral_metodo1', 'tresh1_canny', 'tresh2_canny']
    parametros_trackbar = [[3,49], [1,50],[1,30000],[1,50000], [0,2], [0,255], [0,255], [0,255]]
    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)
    
    variables_trackbar = ['block_size','C','area_min','area_max']
    parametros_trackbar = [[1,49], [1,50],[1,30000],[1,30000]]
    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)
#Función para dibujar círculos en la imagen (con esto obtenemos parámetros)
def encontrar_circulos_hough(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagen_salida = imagen_gris.copy()
        dp = valores_trackbar[0]/10
        minDist = valores_trackbar[1]
        param1 = valores_trackbar[2]
        param2 = valores_trackbar[3]
        minRadius = valores_trackbar[4]
        maxRadius = valores_trackbar[5]
        imagen_gris = cv.GaussianBlur(imagen_gris, (9,9), 4)
        circulos = cv.HoughCircles(imagen_gris, cv.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for i in circulos[0,:]:
                # draw the outer circle
                cv.circle(imagen_salida,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(imagen_salida,(i[0],i[1]),2,(0,0,255),3)
    
        return imagen_salida

    variables_trackbar = ['dp', 'minDist', 'tresh_canny', 'menos_circulos', 'minRadius', 'maxRadius']
    parametros_trackbar = [[10,20],[1,1000], [1, 255], [1, 1000], [1, 1000], [1, 1000]]
    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)
def mostrar_imagen_gris(imagen):
    plt.imshow(imagen,cmap='gray')
    plt.axis('on')
    plt.grid(True)
    plt.show()
def reconstruir_imagen(imagen_erosionada, imagen_original_umbralizada):
    reconstruido = morphology.reconstruction(imagen_erosionada,imagen_original_umbralizada,method='dilation')
    return reconstruido.astype(np.uint8)

def encontrar_contornos(imagen_BGR, umbral_canny0, umbral_canny1):
    imagen_gris = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2GRAY)
    bordes = cv.Canny(imagen_gris, umbral_canny0, umbral_canny1)
    contornos, _ = cv.findContours(bordes, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contornos
def obtener_region_contorno(imagen_gris,contorno):
    x, y, w, h = cv.boundingRect(contorno)
    barcode_region = imagen_gris[y:y+h, x:x+w]
    return barcode_region

def dibujar_contorno(imagen, contorno):
    cv.drawContours(imagen, [contorno], -1, (0, 255, 0), 2)#-1: se dibujan todos, #color (0,255,0) [verde], # trichness = 2

def trackbar_threshold(imagen):
    variables_trackbar = ['umbral_inferior']
    parametros_trackbar = [[0,255]]
    def transformacion(imagen, valores_trackbar):
        umbral_inf = valores_trackbar[0]
        treshold = cv.threshold(imagen, umbral_inf, 255, cv.THRESH_BINARY)[1]
        return treshold
    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)
def trackbar_canny(imagen): 
    variables_trackbar = ['umbral min','umbral max','L2gradiente']
    parametros_trackbar = [[0,255],[0,255],[0,1]]

    def transformacion(imagen, valores_trackbar):
        umbral_minimo = valores_trackbar[0]
        umbral_maximo = valores_trackbar[1]
        L2gradient = valores_trackbar[2]
        gradiente = False
        if L2gradient == 1:
            gradiente = True
        else:
            gradiente = False
        canny = cv.Canny(imagen, umbral_minimo, umbral_maximo, gradiente)
        return canny 

    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)

def trackbar_segmentacion_hsv_inverso(imagen):
    variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255]]

    trackbar_transformacion(imagen, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_hsv_trackbar_mascara_inversa)
    
def trackbar_segmentacion_hsv(imagen):
    variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255]]

    trackbar_transformacion(imagen, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_hsv_trackbar)

def segmentacion_RGB_trackbar(imagen_BGR,valores_trackbar):
    R = valores_trackbar[0]
    G = valores_trackbar[1]
    B = valores_trackbar[2]
    color = (R,G,B)
    radio = valores_trackbar[3]

    imagen_rgb = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2RGB)
    centroide = np.array(color)
    distancias = np.linalg.norm(imagen_rgb - centroide, axis=-1)

    mascara_rgb = np.where(distancias <= radio, 1, 0).astype(np.uint8)
    

    imagen_bgr_segmentada = cv.bitwise_and(imagen_BGR, imagen_BGR, mask=mascara_rgb)    
    
    return imagen_bgr_segmentada#, mascara_rgb
def trackbar_transformacion_RGB(imagen):
    variables_trackbar = ['R', 'G', 'B', 'radio']
    parametros_trackbar = [[0,255],[0,255],[0,255],[0,255]]
    
    trackbar_transformacion(imagen, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_RGB_trackbar)
    
def umbralizado(imagen, valores_trackbar):
    umbral_bajo = valores_trackbar[0]
    _,umbralizado = cv.threshold(imagen, umbral_bajo, 255, cv.THRESH_BINARY)
    return umbralizado

def trackbar_umbral(imagen_BGR):
    imagen_gris = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2GRAY)
    variables_trackbar = ['umbral_bajo','umbral_alto']
    parametros_trackbar = [[0,255],[0,255]]
    
    trackbar_transformacion(imagen_gris, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = umbralizado)
def umbralizado_inv(imagen,valores_trackbar):
    umbral = valores_trackbar[0]
    _,umbralizado = cv.threshold(imagen, umbral, 255, cv.THRESH_BINARY_INV)
    return umbralizado
def trackbar_umbral_inv(imagen_BGR):
    imagen_gris = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2GRAY)
    variables_trackbar = ['umbral']
    parametros_trackbar = [[0,255]]
    
    trackbar_transformacion(imagen_gris, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = umbralizado_inv)

def detectar_lineas_hough(bordes_canny, treshold=100, minima_longitud_linea=50, maximo_gap=10):
    lineas = cv.HoughLinesP(bordes_canny, 1, np.pi/180, treshold, minLineLength=minima_longitud_linea, maxLineGap=maximo_gap)
    lineas_dectadas = []
    if lineas is not None:
        for line in lineas:
            for x1, y1, x2, y2 in line:
                lineas_dectadas.append((x1, y1, x2, y2))

    return lineas_dectadas

def encontrar_intersecciones(bordes_canny, treshold=100, minima_longitud_linea=50, maximo_gap=10):
    lineas = cv.HoughLinesP(bordes_canny, 1, np.pi/180, treshold, minLineLength=minima_longitud_linea, maxLineGap=maximo_gap)
    print(lineas.shape)
    if lineas is not None:
        intersecciones = []
        for i in range(len(lineas)):
            for j in range(i+1, len(lineas)):
                interseccion = calcular_interseccion_otro(lineas[i][0], lineas[j][0])
                if interseccion is not None:
                    intersecciones.append(interseccion)

    return intersecciones

def trackbar_sobel(imagen):
    def transformacion(imagen, valores_trackbar):
        tipo_dato = valores_trackbar[0]
        if tipo_dato == 1:
            ddepth = cv.CV_8U
        else:
            ddepth = cv.CV_64F
        dx = valores_trackbar[1]
        dy = valores_trackbar[2]
        ksize = valores_trackbar[3]
        if ksize % 2 == 0:
            ksize += 1
        dx = np.max(dx,0)
        dy = np.max(dy,0)
        if dx + dy == 0:
            dx += 1
        salida = cv.Sobel(imagen, ddepth, dx, dy, ksize)
        return salida
    variables_trackbar = ['tipo_dato', 'dx', 'dy', 'ksize']
    parametros_trackbar = [[1,2],[0,1],[0,1],[1,31]]
    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)
    
def dibujar_lineas(imagen, lineas):
    for line in lineas:
        cv.line(imagen, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    return imagen

def calcular_interseccion_otro(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    # Calcular los determinantes
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Las líneas son paralelas

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    return (int(px), int(py))

# def calcular_interseccion(linea1, linea2):
#     rho1, theta1 = linea1
#     rho2, theta2 = linea2

#     a1 = np.cos(theta1)
#     b1 = np.sin(theta1)
#     x0_1 = a1 * rho1
#     y0_1 = b1 * rho1

#     a2 = np.cos(theta2)
#     b2 = np.sin(theta2)
#     x0_2 = a2 * rho2
#     y0_2 = b2 * rho2

#     A = np.array([
#         [a1, b1],
#         [a2, b2]
#     ])

#     b = np.array([
#         [x0_1],
#         [x0_2]
#     ])

#     if np.linalg.det(A) == 0:
#         return None #Líneas paralelas

#     x, y = np.linalg.solve(A, b)

#     return int(x), int(y)

def dibujar_intersecciones(imagen, intersecciones):
    for point in intersecciones:
        cv.circle(imagen, point, 5, (0, 0, 255), -1)

def find_countours_intersecciones(img,intersecciones):
    intersections = np.array(intersecciones, dtype=np.int32)
    
    # Imagen vacía
    contour_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Dibuja los puntos como contornos?
    for point in intersections:
        cv.circle(contour_img, point, 1, (255), -1)
    
    contours, _ = cv.findContours(contour_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def llenar_poligonos(imagen, contornos, color=(0,255,0)):
    imagen_salida = imagen.copy()
    for contorno in contornos:
        poligono_convexo = cv.convexHull(contorno)
        cv.fillConvexPoly(imagen_salida, poligono_convexo,color)
    return imagen_salida
###########################################################################################################################
#                                                 HOUGH
###########################################################################################################################
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

def tranformada_fourier_imagen(imagen):
    fft_imagen = fft2(imagen)
    fft_imagen_shifted = fftshift(fft_imagen)
    magnitude_spectrum = 20 * np.log(np.abs(fft_imagen_shifted) + 1)
    return magnitude_spectrum

def tranformada_inversa_fourier(imagen_fft_shifteada):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(imagen_fft_shifteada)))

def fft_y_shift(imagen):
    fft_imagen = fft2(imagen)
    fft_imagen_shifted = fftshift(fft_imagen)
    return fft_imagen_shifted

# Función para calcular el ángulo de una línea
def calcular_angulo_linea(linea):
    x1, y1, x2, y2 = linea[0]
    angulo = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angulo

def rotar_imagen(imagen):
    #calcular ángulo
    angulo = calcular_angulo_linea(imagen)
    imagen_rotada = rotate(imagen, angulo)
    return imagen_rotada

def calcular_angulo_rotacion(imagen_original, imagen_rotada, umbral=190, umbral_hough = 180,
                                          min_line_length = 280, max_line_gap = 2):
    # Transformada de Fourier y normalización
    imagen_rotada_fft_magnitud = tranformada_fourier_imagen(imagen_rotada)
    imagen_original_fft_magnitud = tranformada_fourier_imagen(imagen_original)
    imagen_original_fft_magnitud = cv.normalize(imagen_original_fft_magnitud, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    imagen_rotada_fft_magnitud = cv.normalize(imagen_rotada_fft_magnitud, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Detección de bordes con Canny
    bordes_original = cv.Canny(imagen_original_fft_magnitud, umbral, 255)
    bordes_rotada = cv.Canny(imagen_rotada_fft_magnitud, umbral, 255)

    # Detección de líneas con HoughLinesP
    lineas_originales = cv.HoughLinesP(bordes_original, 1, np.pi / 180,
                                        umbral_hough, minLineLength=min_line_length, maxLineGap=max_line_gap)
    lineas_rotada = cv.HoughLinesP(bordes_rotada, 1, np.pi*2 / 180,
                                    umbral_hough, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lineas_originales is None or lineas_rotada is None:
        return 0
    # Calcular ángulos de las líneas
    angulos_originales = [calcular_angulo_linea(linea) for linea in lineas_originales]
    angulos_rotada = [calcular_angulo_linea(linea) for linea in lineas_rotada]

    # Promediar los ángulos para obtener el ángulo dominante
    angulo_promedio_original = np.median(angulos_originales)
    angulo_promedio_rotada = np.median(angulos_rotada)


    # Calcular el ángulo de rotación necesario para alinear las imágenes
    angulo_rotacion = angulo_promedio_rotada - angulo_promedio_original

    return angulo_rotacion

def calcular_angulo_y_rotar(imagen_original, imagen_rotada, umbral=190, umbral_hough = 180,
                                          min_line_length = 280, max_line_gap = 2):
    angulo_rotacion = calcular_angulo_rotacion(imagen_original, imagen_rotada, umbral=umbral, 
                                                            umbral_hough = umbral_hough, min_line_length = min_line_length, max_line_gap = max_line_gap)
    print(angulo_rotacion)
    imagen_rotada_corregida = imutils.rotate(imagen_rotada, angulo_rotacion)

    return imagen_rotada_corregida

def calcular_angulo_y_rotar_una_imagen(imagen_original, umbral=190, umbral_hough = 180,
                                          min_line_length = 280, max_line_gap = 2):
    angulo_rotacion = detectar_rotacion_una_imagen(imagen_original, umbral=umbral, 
                                                            tresh_hough= umbral_hough, min_line_length = min_line_length, max_line_gap = max_line_gap)
    print(angulo_rotacion)
    imagen_rotada_corregida = imutils.rotate(imagen_original, angulo_rotacion)

    return imagen_rotada_corregida

def trackbar_angulo_rotacion(imagen_base, imagen_rot):
    def transformacion(imagen_rot,valores_trackbar):
        umbral = valores_trackbar[0]
        umbral_hough = valores_trackbar[1]
        min_line_length = valores_trackbar[2]
        max_line_gap = valores_trackbar[3]
        imagen_salida = calcular_angulo_y_rotar(imagen_base, imagen_rot, umbral=umbral, umbral_hough = umbral_hough,
                                          min_line_length = min_line_length, max_line_gap = max_line_gap)
    
        return imagen_salida
    variables_trackbar = ['umbral','umbral_hough','min_line_length','max_line_gap']
    parametros_trackbar = [[0,255],[0,255],[0,255],[0,1000],[0,1000]]
    trackbar_transformacion_val(imagen_rot, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = transformacion)
def detectar_angulo_rotacion_dos_imagenes(imagen_base,imagen_rot, umbral=190, umbral_hough = 180,
                                          min_line_length = 280, max_line_gap = 2):
    imagen_base_gris = cv.cvtColor(imagen_base, cv.COLOR_BGR2GRAY)
    imagen_rot_gris = cv.cvtColor(imagen_rot, cv.COLOR_BGR2GRAY)

    espectro_base = np.abs(fft_y_shift(imagen_base_gris))
    _, espectro_base = cv.threshold(espectro_base,umbral,255,cv.THRESH_BINARY)
    lines = cv.HoughLinesP(espectro_base.astype(np.uint8), 1, np.pi/180, 
                           umbral_hough, minLineLength=min_line_length, maxLineGap=max_line_gap)
    xbase1, ybase1, xbase2, ybase2 = lines[0][0]
    cv.line(espectro_base, (xbase1, ybase1), (xbase2, ybase2), (0, 255, 0), 2)
    # plt.figure(),plt.imshow(espectro_base),plt.title('sin rotar')

    punto_base1=np.array([xbase1, ybase1])  
    punto_base2=np.array([xbase2, ybase2])
    
    vector_base=punto_base2-punto_base1

    #DATOS DE LA IMAGEN ROTADA
    espectro_rotado = np.abs(fft_y_shift(imagen_rot_gris))
    ret,espectro_rotado = cv.threshold(espectro_rotado,umbral,255,cv.THRESH_BINARY)
    lines = cv.HoughLinesP(espectro_rotado.astype(np.uint8), 1, np.pi/180,
                            umbral_hough, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return 0
    
    x1, y1, x2, y2 = lines[0][0]

    cv.line(espectro_rotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Para visualizar
    # plt.figure(),plt.imshow(espectro_rotado),plt.title('rotado')

    #calculo el vector de la imagen rotada
    punto_rot1=np.array([x1, y1])
    punto_rot2=np.array([x2, y2])
    vector_rot=punto_rot2-punto_rot1

    Angulo=math.acos(np.dot(vector_base, vector_rot) / (np.linalg.norm(vector_base) * np.linalg.norm(vector_rot)))
    Angulo=Angulo*180/np.pi

    return Angulo

def aplicar_hough_probabilistico(imagen, umbrales_canny : tuple[int,int], rho=1, theta=np.pi/180,umbral_hough=200, minLineLength=100, maxLineGap=10):
    imagen_resultado = imagen.copy()
    umbral0 = umbrales_canny[0]
    umbral1 = umbrales_canny[1]
    bordes = cv.Canny(imagen, umbral0, umbral1)
    lineas = cv.HoughLinesP(bordes, rho, theta, umbral_hough, minLineLength, maxLineGap)
    angulos = calcular_angulos_rotacion(lineas)
    return lineas, angulos

def rotate(img, angle):
    """Rotación de la imagen sobre el centro"""
    r = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1.0)
    # The corrected line is below
    return cv.warpAffine(img, r, (img.shape[1], img.shape[0]))

# Este mejor si solo la imagen es bien limpia
# def extraer_ROI_mascara(mascara):
#     # Dada una imagen binaria (máscara)
#     y_indices, x_indices = np.nonzero(mascara)
#     x_min, x_max = x_indices.min(), x_indices.max()
#     y_min, y_max = y_indices.min(), y_indices.max()
#     roi = mascara[y_min:y_max+1, x_min:x_max+1]
#     return roi

#Este anda mejor imagenes que tienen porquería 
def extraer_ROI_mascara(mascara):
    rows, cols = mascara.shape
    top_left = (0, 0)
    bottom_right = (cols, rows)

    contours, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(largest_contour)

    top_left_roi = (x, y)
    bottom_right_roi = (x + w, y + h)

    roi = mascara[top_left_roi[1]:bottom_right_roi[1], top_left_roi[0]:bottom_right_roi[0]]

    return roi

def trackbar_angulo_rotacion_una_imagen(imagen):
    def transformacion(imagen,valores_trackbar):
        umbral = valores_trackbar[0]
        umbral_hough = valores_trackbar[1]
        min_line_length = valores_trackbar[2]
        max_line_gap = valores_trackbar[3]
        imagen_salida = calcular_angulo_y_rotar_una_imagen(imagen, umbral=umbral, umbral_hough= umbral_hough,
                                          min_line_length = min_line_length, max_line_gap = max_line_gap)
    
        return imagen_salida
    variables_trackbar = ['umbral','umbral_hough','min_line_length','max_line_gap']
    parametros_trackbar = [[0,255],[0,255],[0,255],[0,1000],[0,1000]]
    trackbar_transformacion_val(imagen, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = transformacion)
    
def detectar_rotacion_una_imagen(imagen, umbral=50, tresh_hough = 100, min_line_length = 100, max_line_gap = 10): #esto anda pero ni idea
    edges = cv.Canny(imagen, umbral, 255)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=tresh_hough,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    mean_angle = np.median(angles) #Anda mejor (El promedio te hace perder precisión)

    # Adjust the angle to the range [-90, 90]
    # rotation_angle = (median_angle - 90) if median_angle > 90 else median_angle
    # return rotation_angle

    return mean_angle
###########################################################################################################################
#                                                  FUNCIONES GENERALES
########################################################################################################################
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
###########################################################################################################################
#                                                  COLOR
###########################################################################################################################

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

def extraer_HSI(imagen):
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    hue, saturation, _ = cv.split(imagen_hsv)
    rojo, verde, azul = cv.split(imagen)
    intensidad = (rojo+verde+azul)/3
    return hue, saturation, intensidad

###########################################################################################################################
#                                                  HISTOGRAMAS
###########################################################################################################################

def calcular_histogramas(imagenes):
    """
    Calcula los histogramas para un arreglo de imágenes.

    Args:
        imagenes (list): Lista de imágenes (cada imagen como un arreglo numpy).

    Returns:
        list: Lista de histogramas (cada histograma como un arreglo numpy).
    """
    histogramas = []
    for img in imagenes:
        # Convierte la imagen a escala de grises si es a color
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Calcula el histograma
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        histogramas.append(hist)
    return histogramas

def graficar_histogramas_subplots(histogramas):
    """
    Grafica los histogramas en subplots.

    Args:
        histogramas (list): Lista de histogramas (cada histograma como un arreglo numpy).
    """
    num_imagenes = len(histogramas)
    fig, axs = plt.subplots(1, num_imagenes, figsize=(6*num_imagenes, 6))

    for i, hist in enumerate(histogramas):
        axs[i].bar(range(256), hist, width=1, edgecolor='none')
        axs[i].set_xlim(-5, 260)
        axs[i].set_title(f"Histograma de Imagen {i+1}")
        axs[i].set_xlabel("Intensidad de píxeles")
        axs[i].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

###########################################################################################################################
#                                                  FILTROS
###########################################################################################################################



def filtro_pasa_bajos_ideal(size, D0):
    filas, columnas = size
    centro_fila = filas // 2
    centro_columna = columnas // 2
    filtro = np.zeros((filas, columnas), np.float32)
    centro = (filas//2,columnas//2)
    for i in range(filas):
        for j in range(columnas):
            if np.sqrt((i - centro_fila)**2 + (j - centro_columna)**2) <= D0:
                filtro[i, j] = 1
    return filtro

def aplicar_filtro(imagen, filtro): #Recibe f(x,y) y H(u,v)
    fft_imagen = np.fft.fftshift(np.fft.fft2(imagen)) #F(u,v)
    fft_filtro = filtro #H(u,v)
    fft_filtrada = fft_imagen * fft_filtro #F(u,v)*H(u,v) => mult. en frecuencia, convoluciono en espacio
    imagen_filtrada = np.fft.ifft2(np.fft.ifftshift(fft_filtrada)).real
    return imagen_filtrada

def filtro_butterworth_pasa_bajos(shape, D0, n):
    filas, columnas = shape
    filtro = np.zeros((filas, columnas), np.float32)
    centro_fila = filas // 2
    centro_columna = columnas // 2
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - centro_fila)**2 + (j - centro_columna)**2)
            filtro[i, j] = 1 / (1 + (distancia / D0)**(2 * n))
    return filtro

def filtro_gaussiano_pasa_bajos(shape, sigma):
    filas, columnas = shape
    centro_fila = filas // 2
    centro_columna = columnas // 2
    
    filtro = np.zeros((filas, columnas))
    
    for x in range(filas):
        for y in range(columnas):
            filtro[x, y] = np.exp(-((x - centro_fila)**2 + (y - centro_columna)**2) / (2 * sigma**2))
    
    filtro /= np.sum(filtro)
    
    return filtro

def filtro_pasa_altos_ideal(size, D0):
    filas, columnas = size
    centro_fila = filas // 2
    centro_columna = columnas // 2
    filtro = np.ones((filas, columnas), np.float32)
    for i in range(filas):
        for j in range(columnas):
            if np.sqrt((i - centro_fila)**2 + (j - centro_columna)**2) <= D0:
                filtro[i, j] = 0
    return filtro

def filtro_butterworth_pasa_altos(size, D0, n):
    filas, columnas = size
    filtro = np.zeros((filas, columnas), np.float32)
    centro_fila = filas // 2
    centro_columna = columnas // 2
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - centro_fila)**2 + (j - centro_columna)**2) + 1e-6
            filtro[i, j] = 1 / (1 + (D0 / distancia)**(2 * n))
    return filtro

def filtro_gaussiano_pasa_altos(size, sigma):
    centro = size // 2
    
    filtro = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):
            filtro[x, y] = 1 - np.exp(-((x - centro)**2 + (y - centro)**2) / (2 * sigma**2))
    
    filtro /= np.sum(filtro)
    
    return filtro


def filtro_pasa_altos(imagen):
    
    mascara_cruz = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]], dtype=np.float32)
    
    imagen_filtrada = cv.filter2D(imagen, -1, mascara_cruz)

    return imagen_filtrada

def filtro_alta_potencia(imagen_pasa_altos, A):

    H_alta_potencia = (A - 1) + imagen_pasa_altos

    return H_alta_potencia

def filtro_enfasis_alta_frecuencia(imagen_pasa_altos, a, b):
                                 
    
    H_enfasis_alta_frecuencia = a + b * imagen_pasa_altos

    return H_enfasis_alta_frecuencia

###########################################################################################################################
#                                                  OTRAS POSIBLES COSAS ÚTILES
###########################################################################################################################

def plotear_planos_de_bits(imagen_gris):
    # Obtener las dimensiones de la imagen
    filas, columnas = imagen_gris.shape
    
    # Crear una figura con un tamaño adecuado para mostrar todos los planos de bits
    plt.figure(figsize=(8, 6))
    
    # Iterar sobre cada bit
    for i in range(8):
        # Crear una matriz de ceros para almacenar el plano de bits
        plano_de_bits = np.zeros((filas, columnas), dtype=np.uint8)
        
        # Iterar sobre cada píxel de la imagen
        for fila in range(filas):
            for columna in range(columnas):
                # Obtener el valor del bit en la posición 'i' del píxel
                bit = (imagen_gris[fila, columna] >> i) & 1
                # Asignar el valor del bit al plano de bits en la misma posición
                plano_de_bits[fila, columna] = bit
        
        # Plotear el plano de bits en un subplot
        plt.subplot(2, 4, i+1)  # Filas: 2, Columnas: 4, Posición: i+1
        plt.imshow(plano_de_bits, cmap='gray')
        plt.title(f'Bit {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def bounding_box(imagen):
    indices_objeto = np.where(imagen == 255)

    # Encuentra los límites superior, inferior, izquierdo y derecho del objeto
    top = np.min(indices_objeto[0])
    bottom = np.max(indices_objeto[0])
    left = np.min(indices_objeto[1])
    right = np.max(indices_objeto[1])
    print(left,right,top,bottom)
    # esquina_superior_izquierda = [left,top]
    # esquina_inferior_derecha = [right,bottom]

    return [left,right,top,bottom]#Izquierda, derecha, arriba, abajo

def nothing(x):
    pass
def mostrar_grafico_trackbar(imagen_original):
  NOMBRE_VENTANA = "IMAGEN TRANSFORMADA"
  cv.namedWindow(NOMBRE_VENTANA)

  cv.createTrackbar("a", NOMBRE_VENTANA,10,100,nothing)
  cv.createTrackbar("c", NOMBRE_VENTANA,0,200,nothing)

  # img = cv2.imread('imagenes_varias/micky.jpg',cv2.IMREAD_GRAYSCALE)
  # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

  while(1):
    a = cv.getTrackbarPos("a", NOMBRE_VENTANA)
    c = cv.getTrackbarPos("c", NOMBRE_VENTANA)

    #  imagen_salida = np.clip(a * img + c, 0, 255).astype(np.uint8)
    imagen_salida = cv.convertScaleAbs((a/10)*imagen_original+(c-100))

    cv.imshow("Imagen resultado",imagen_salida)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
      break
  cv.destroyAllWindows()

def mostrar_imagenes(imagenes):
    num_imagenes = len(imagenes)
    fig1, axs = plt.subplots(1, num_imagenes, figsize=(15, 5))
    
    for i, imagen in enumerate(imagenes):
        imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)  # Convertir la imagen a RGB para Matplotlib
        axs[i].imshow(imagen_rgb)
        axs[i].axis('off')  # Ocultar ejes
        
    plt.show()


def evitar_desborde(imagen):

    minimo = np.min(imagen)
    maximo = np.max(imagen)

    if minimo < 0:
        imagen = imagen + 255
        imagen = imagen / 2
    if maximo > 255:
        imagen = (imagen - minimo)*(255/(maximo-minimo))
    
    return imagen
def diferencia(imagen1,imagen2):
    diferencia_imagenes = imagen1 - imagen2
    return evitar_desborde(diferencia_imagenes)

def contar_hasta_cero(arreglo):
    elementos_hasta_cero = 0

    for elemento in arreglo:
        if elemento != 0:
            elementos_hasta_cero += 1
        else:
            break

    return elementos_hasta_cero
def contar_valores_seguidos(arreglo, valor):
    cantidad_valores_seguidos = 0
    valores_actuales = 0

    for elemento in arreglo:
        if elemento == valor:
            valores_actuales += 1
            cantidad_valores_seguidos = max(cantidad_valores_seguidos, valores_actuales)
        else:
            valores_actuales = 0

    return cantidad_valores_seguidos
def contar_ceros_y_no_ceros_seguidos(arreglo):
    cantidad_ceros_seguidos = 0
    cantidad_no_ceros_seguidos = 0
    ceros_actuales = 0
    no_ceros_actuales = 0

    for elemento in arreglo:
        if elemento == 0:
            ceros_actuales += 1
            cantidad_no_ceros_seguidos = max(cantidad_no_ceros_seguidos, no_ceros_actuales)
            no_ceros_actuales = 0
        else:
            no_ceros_actuales += 1
            cantidad_ceros_seguidos = max(cantidad_ceros_seguidos, ceros_actuales)
            ceros_actuales = 0

    cantidad_ceros_seguidos = max(cantidad_ceros_seguidos, ceros_actuales)
    cantidad_no_ceros_seguidos = max(cantidad_no_ceros_seguidos, no_ceros_actuales)

    return cantidad_ceros_seguidos, cantidad_no_ceros_seguidos

def trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion):
    imagen_original = imagen.copy()
    NOMBRE_VENTANA = 'Trackbars'
    def on_trackbar(value):
        # valores_trackbar = []
        # for i in range(len(variables_trackbar)):
        #     valores_trackbar.append(cv.getTrackbarPos(variables_trackbar[i], NOMBRE_VENTANA))
        valores_trackbar = [cv.getTrackbarPos(var, NOMBRE_VENTANA) for var in variables_trackbar]
        
        imagen_transformada = transformacion(imagen_original, valores_trackbar)

        imagen_transformada_normalizada = cv.normalize(imagen_transformada,None,0,255,cv.NORM_MINMAX)

        imagen_resultado = cv.convertScaleAbs(imagen_transformada_normalizada)

        cv.imshow('Imagen Original', imagen)
        cv.imshow('Imagen Transformada', imagen_resultado)

    cv.namedWindow(NOMBRE_VENTANA)

    for i, var in enumerate(variables_trackbar):
        cv.createTrackbar(var, NOMBRE_VENTANA, parametros_trackbar[i][0], parametros_trackbar[i][1], on_trackbar)

    on_trackbar(0)

    cv.waitKey(0)
    cv.destroyAllWindows()

def calcular_snr(imagen):
    if len(imagen.shape) == 3:
        imagen = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    signal_power = np.mean(imagen) ** 2

    noise_power = np.var(imagen)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

#-Función que a partir de una imagen y valores de trackbars genera una imagen transforamada
# variables_trackbar: se ponen las variables a poner en las trackbars
# parámetros_trackbar: se ponen los parámetros de cada variable de las trackbars
# si no se pone nada solo muestra 
# transformacion: funcion que se va a aplicar con los valores de las trackbars
def ventana_trackbars(imagen, variables_trackbar = None, parametros_trackbar = None, transformacion = None):
  
  NOMBRE_VENTANA = "IMAGEN_y_TRANSFORMADA"
  cv.namedWindow(NOMBRE_VENTANA)

  for i in range(len(variables_trackbar)):
    bordes = parametros_trackbar[i]
    cv.createTrackbar(variables_trackbar[i], NOMBRE_VENTANA,bordes[0],bordes[1],nothing)

  # Hago un bluce infinito donde tomo valores de las trackbars
  #valores_trackbars = np.zeros((1,len(variables_trackbar)))
  while True:

    valores_trackbars = []
    for i in range(len(variables_trackbar)):
        valores_trackbars.append(cv.getTrackbarPos(variables_trackbar[i], NOMBRE_VENTANA))
    
    imagen_salida = cv.convertScaleAbs(transformacion(imagen, valores_trackbars))
    
    imagen_combinada = np.hstack((imagen,imagen_salida))

    cv.imshow(NOMBRE_VENTANA,imagen_combinada)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
      break
  cv.destroyAllWindows()


def mostrar_imagen_binaria(image):
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()


def mostrar_fft(fft_image):
    #np.log(np.abs(fft_image) + 1) #log para comprimir valores, +1 para evitar valores cercanos a 0 y abs para quitar complejos
    plt.imshow(np.log(np.abs(fft_image) + 1), cmap='viridis', interpolation='nearest')
    plt.axis('off')
    plt.colorbar(label='Log Magnitude')
    plt.show()

def calcular_fft(binary_image):
    fft_image = fftshift(fft2(binary_image)) #fft2 calcula la transformada 2d y fftshift la centra
    return fft_image


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

#####################################################################################################
#                                             RUIDOS
#####################################################################################################
def ruido_gaussiano(shape, media=0, desviacion_estandar=10):
    fila, columna = shape
    
    ruido_gaussiano = np.random.normal(media, desviacion_estandar, (fila, columna))
    
    ruido_gaussiano = ruido_gaussiano - np.mean(ruido_gaussiano)
    return ruido_gaussiano

def ruido_uniforme(shape, bajo=-20, alto=20):
    fila, columna = shape
    
    ruido_uniforme = np.random.uniform(bajo, alto, (fila, columna))
    ruido_uniforme = ruido_uniforme - np.mean(ruido_uniforme)
    
    return ruido_uniforme

def ruido_sal_y_pimienta(imagen, cantidad_max=20000):
    fila, columna = imagen.shape
    imagen_ruidosa = np.copy(imagen)
    numero_de_pixeles = np.random.randint(10000,cantidad_max)

    for _ in range(numero_de_pixeles):
        x = np.random.randint(0, fila-1)
        y = np.random.randint(0, columna-1)
        imagen_ruidosa[x, y] = np.random.choice([0, 255])

    return imagen_ruidosa

def ruido_impulsivo_unimodal(shape, prob=0.05, valor=0.1):
    filas, columnas = shape
    
    imagen_ruidosa = np.zeros(shape, dtype=np.uint8)
    
    mascara_ruido = np.random.rand(filas,columnas) 

    imagen_ruidosa[np.where(mascara_ruido < prob)] = valor

    return imagen_ruidosa

def ruido_exponencial(shape, escala=10):
    fila, columna = shape
    
    ruido_exponencial = np.random.exponential(escala, (fila, columna))
    
    ruido_exponencial = ruido_exponencial - np.mean(ruido_exponencial)

    return ruido_exponencial

#####################################################################################################
#                                             RUIDOS
#####################################################################################################

#Función findCountours
#mode: Modo de recuperación de contornos. Algunas opciones comunes son:
    # cv.RETR_EXTERNAL: Recupera solo los contornos externos.
    # cv.RETR_LIST: Recupera todos los contornos y no crea ninguna jerarquía.
    # cv.RETR_CCOMP: Recupera todos los contornos y organiza una jerarquía de contornos de dos niveles.
    # cv.RETR_TREE: Recupera todos los contornos y organiza en una jerarquía completa de contornos anidados.
#method: Método de aproximación de contornos. Algunas opciones comunes son:
    # cv.CHAIN_APPROX_NONE: Guarda todos los puntos del contorno.
    # cv.CHAIN_APPROX_SIMPLE: Comprime los segmentos horizontales, verticales y diagonales, y deja solo sus puntos finales. Por ejemplo, un rectángulo se representará solo con cuatro puntos.
# Devuelve contours (lista de contornos encontrados (cada uno un array de puntos [coordenadas x e y]) y 
#hierarchy que es de la misma longitud que countours y contiene información sobre la imagen topológica )


# Función cv.erode()
    #border_type: 
    # cv2.BORDER_CONSTANT: Pads the image with a constant value (specified by borderValue).
    # cv2.BORDER_REPLICATE: Replicates the last pixel in the border.
    # cv2.BORDER_REFLECT: Reflects the border pixels.
    # cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT: Reflects the border pixels with a slight shift.
    # cv2.BORDER_WRAP: Wraps around the border pixels.

    # border_value: 
    # This parameter is used in conjunction with cv2.BORDER_CONSTANT to specify the value to be used for padding the image.
    # The value should be a scalar or a tuple corresponding to the number of image channels