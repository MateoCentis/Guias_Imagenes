import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.fft import fftshift, fft2
import os
import math
from skimage import morphology

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
def rectangulos_trackbar(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagen_salida = imagen.copy()
        block_size = valores_trackbar[0]
        if block_size % 2 == 0:
            block_size = block_size + 1
        C = valores_trackbar[1]
        area_min = valores_trackbar[2]
        area_max = valores_trackbar[3]
        metodo = valores_trackbar[4]
        umbral = valores_trackbar[5]
        if metodo == 0:
            imagen_bin = cv.adaptiveThreshold(imagen_gris, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)
        elif metodo == 1:
            imagen_bin = cv.threshold(imagen_gris, umbral, 255, cv.THRESH_BINARY)[1]
        else:
            imagen_bin = cv.threshold(imagen_gris, umbral, 255, cv.THRESH_BINARY_INV)[1]
        contornos,_ = cv.findContours(imagen_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            x, y, w, h = cv.boundingRect(contorno)
            area = w*h
            if area_min <= area <= area_max:
                cv.rectangle(imagen_salida, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return imagen_salida
    
    variables_trackbar = ['block_size','C','area_min','area_max','metodo_umbral','umbral']
    parametros_trackbar = [[3,49], [1,50],[1,30000],[1,30000],[0,2],[0,255]]
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

    variables_trackbar = ['dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius']
    parametros_trackbar = [[10,20],[1,1000], [1, 1000], [1, 1000], [1, 1000], [1, 1000]]
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
    umbral_alto = valores_trackbar[1]
    _,umbralizado = cv.threshold(imagen, umbral_bajo, umbral_alto, cv.THRESH_BINARY)
    return umbralizado

def trackbar_umbral(imagen_BGR):
    imagen_gris = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2GRAY)
    variables_trackbar = ['umbral_bajo','umbral_alto']
    parametros_trackbar = [[0,255],[0,255]]
    
    trackbar_transformacion(imagen_gris, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = umbralizado)
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