import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.fft import fftshift, fft2

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

def filtro_gaussiano_pasa_bajos(size, sigma):
    centro = size // 2
    
    filtro = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):
            filtro[x, y] = np.exp(-((x - centro)**2 + (y - centro)**2) / (2 * sigma**2))
    
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
