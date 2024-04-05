import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


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

# Ejemplo de uso
# Supongamos que tenemos un arreglo 'imagenes' con las imágenes cargadas
# (puedes reemplazarlo con tus propias imágenes)
# histogramas = calcular_histogramas(imagenes)
# graficar_histogramas_subplots(histogramas)


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

# def mostrar_imagenes(rutas_imagenes):
#     num_imagenes = len(rutas_imagenes)
#     fig, axs = plt.subplots(1, num_imagenes, figsize=(15, 5))
    
#     for i, ruta_imagen in enumerate(rutas_imagenes):
#         imagen = cv.imread(ruta_imagen)
#         imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)  # Convertir la imagen a RGB para Matplotlib
#         axs[i].imshow(imagen_rgb)
#         axs[i].axis('off')  # Ocultar ejes
        
#     plt.show()


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

#-Función que a partir de una imagen y valores de trackbars genera una imagen transforamada
# variables_trackbar: se ponen las variables a poner en las trackbars
# parámetros_trackbar: se ponen los parámetros de cada variable de las trackbars
# si no se pone nada solo muestra 
# transformacion: funcion que se va a aplicar con los valores de las trackbars
def ventana_trackbars(imagen, variables_trackbar = None, parametros_trackbar = None, transformacion = None):
  #Validaciones pavas
  if variables_trackbar:
      if parametros_trackbar:
          if len(variables_trackbar) != len(parametros_trackbar):
              print("Trackbars y parámetros debe tener el mismo tamaño")
              return
      else:
          print("Si hay trackbars tienen que haber parámetros")
          return
  
  # Defino ventana y sus trackbars correspondientes
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
    
    imagen_salida = evitar_desborde(transformacion(imagen, valores_trackbars))
    
    imagen_combinada = np.hstack((imagen,imagen_salida))

    cv.imshow(NOMBRE_VENTANA,imagen_combinada)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
      break
  cv.destroyAllWindows()
