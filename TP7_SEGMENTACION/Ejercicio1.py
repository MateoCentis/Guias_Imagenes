import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2 as cv
from utils import trackbar_transformacion, ruido_impulsivo_unimodal, ruido_gaussiano, ventana_trackbars
def extraer_perfiles_gris(imagen):
    def on_click(event):
        if event.inaxes is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            print("X: ", x)
            print("Y: ", y)
            perfilesX = imagen[x, :]
            perfilesY = imagen[:, y]
            
            plt.figure(figsize=(12, 6))

            # Perfil de intensidad de la fila
            plt.subplot(1, 2, 1)
            plt.plot(perfilesX, color='gray')
            plt.title(f'Perfil de intensidad de la fila {x}')
            plt.xlabel('y')
            plt.ylabel('Intensidad')
            plt.grid(True)

            # Perfil de intensidad de la columna
            plt.subplot(1, 2, 2)
            plt.plot(perfilesY, color='gray')
            plt.title(f'Perfil de intensidad de la columna {y}')
            plt.xlabel('x')
            plt.ylabel('Intensidad')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()

def prewitt_deteccion_bordes(imagen, umbral=0.1):  
    # Kernels de Prewitt para la detección de bordes en x y y
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])

    gradiente_x = convolve2d(imagen, kernel_x, mode='same', boundary='symm')
    gradiente_y = convolve2d(imagen, kernel_y, mode='same', boundary='symm')
    
    # magnitud_gradiente = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitud_gradiente = np.abs(gradiente_x) + np.abs(gradiente_y)
    
    edge_imagen = np.zeros_like(magnitud_gradiente)
    edge_imagen[magnitud_gradiente > umbral] = 1
    
    return edge_imagen

def laplaciano_deteccion_bordes(imagen, ddepth, ksize):
    laplacian_image = cv.Laplacian(imagen, ddepth, ksize=ksize)
    return laplacian_image

def sobel_deteccion_bordes(imagen, ddepth=cv.CV_8U, dx=1, dy=0, ksize=3):
    if ksize == -1:
        sobel_image = cv.Scharr(imagen,ddepth,dx,dy)
    else:
        sobel_image = cv.Sobel(imagen, ddepth, dx, dy, ksize=ksize)
    return sobel_image

imagen = cv.imread("Imagenes_Ej/patron_bordes.jpg", cv.IMREAD_GRAYSCALE)

def canny_deteccion_bordes(imagen, umbral_minimo, umbral_maximo, L2gradient):
    edges = cv.Canny(imagen, umbral_minimo, umbral_maximo, L2gradient=L2gradient)
    return edges

ver_perfiles = False
if ver_perfiles:
    extraer_perfiles_gris(imagen)

filtro_prewitt = False
if filtro_prewitt:
    variables_trackbar = ['umbral']
    parametros_trackbar = [[1,2000]]

    def transformacion(imagen, valores_trackbar):
        umbral = valores_trackbar[0]/100
        bordes_prewitt = prewitt_deteccion_bordes(imagen, umbral)
        return bordes_prewitt

    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)

filtro_sobel = False
if filtro_sobel:
    variables_trackbar = ['Tipo dato','dx','dy','ksize']
    parametros_trackbar = [[1,2],[0,1],[0,1],[1,31]]

    def transformacion(imagen, valores_trackbar):
        tipo_dato = valores_trackbar[0]
        dx = int(np.max(valores_trackbar[1],0))
        dy = int(np.max(valores_trackbar[2],0))
        ksize = valores_trackbar[3]-1
        if tipo_dato == 1:
            ddepth = cv.CV_8U
        else:
            ddepth = cv.CV_64F
        if ksize % 2 == 0:
            ksize += 1
        if dx + dy == 0:
            dx += 1
        sobel = sobel_deteccion_bordes(imagen, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize)
        return sobel

    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)

filtro_laplaciano = False
if filtro_laplaciano:
    variables_trackbar = ['Tipo dato','ksize']
    parametros_trackbar = [[1,2],[1,31]]

    def transformacion(imagen, valores_trackbar):
        tipo_dato = valores_trackbar[0]
        ksize = valores_trackbar[1]-1
        if tipo_dato == 1:
            ddepth = cv.CV_8U
        else:
            ddepth = cv.CV_64F
        if ksize % 2 == 0:
            ksize += 1
        laplaciano = laplaciano_deteccion_bordes(imagen, ddepth=ddepth, ksize=ksize)
        return laplaciano

    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)

filtro_canny = True
if filtro_canny:
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
        canny = canny_deteccion_bordes(imagen, umbral_minimo, umbral_maximo, gradiente)
        return canny 

    trackbar_transformacion(imagen, variables_trackbar, parametros_trackbar, transformacion)

#-------------------------------Agregado de ruido y detección de bordes----------------------------------------
def ruido_impulsivo(imagen, nivel_ruido):
    noisy_image = np.copy(imagen)
    height, width = imagen.shape
    num_pixels_to_corrupt = int(nivel_ruido * height * width)
    indices_to_corrupt = np.random.choice(height * width, num_pixels_to_corrupt, replace=False)
    noisy_pixels = np.random.randint(0, 256, num_pixels_to_corrupt)  # Generar valores de ruido para los píxeles corruptos
    noisy_image[np.unravel_index(indices_to_corrupt, (height, width))] = noisy_pixels  # Aplicar ruido a los píxeles seleccionados
    return noisy_image

parte5 = True
if parte5:
    mosquito = cv.imread("Imagenes_Ej/mosquito.jpg", cv.IMREAD_GRAYSCALE)

    #Variando valor donde aparece
    probabilidad = 0.05 
    valor = 24
    ruido_u = ruido_impulsivo_unimodal(mosquito.shape, probabilidad,valor)
    #variando desviación estandar
    stdv = 10
    ruido_g = ruido_gaussiano(mosquito.shape, 0, stdv)

    variables_trackbar = ['desviacionSTD', 'valor','tipo dato','ksize', 
                          'umbral_min', 'umbral_max', 'L2Grad','dx', 'dy']
    parametros_trackbar = [[0,255],[0,255],[1,2], [1,31], [0,255], [0,255], [0,1],[0,1],[0,1]]
    shape_mosquito = mosquito.shape
    def transformacion(imagen, valores_trackbar):
        global shape_mosquito
        stdv = valores_trackbar[0]
        valor = valores_trackbar[1]
        tipo_dato = valores_trackbar[2]
        ksize = valores_trackbar[3]-1
        if tipo_dato == 1:
            ddepth = cv.CV_8U
        else:
            ddepth = cv.CV_64F
        if ksize % 2 == 0:
            ksize += 1
        umbral_minimo = valores_trackbar[4]
        umbral_maximo = valores_trackbar[5]
        L2gradient = valores_trackbar[6]
        dx = np.max(valores_trackbar[7],0)
        dy = np.max(valores_trackbar[8],0)
        if L2gradient == 1:
            gradiente = True
        else:
            gradiente = False
        if dx + dy == 0:
            dx += 1
        ruido_g = ruido_gaussiano(shape_mosquito, 0, stdv)
        ruido_u = ruido_impulsivo_unimodal(shape_mosquito, 0.05, valor)
        imagen_ruido = (imagen + ruido_g + ruido_u).astype(np.uint8)
        salida = sobel_deteccion_bordes(imagen_ruido, ddepth, dx, dy, ksize)
        # salida = canny_deteccion_bordes(imagen_ruido,umbral_minimo, umbral_maximo, gradiente)
        # salida = laplaciano_deteccion_bordes(imagen_ruido, ddepth, ksize)
        return salida

    ventana_trackbars(mosquito, variables_trackbar, parametros_trackbar, transformacion)
    # variables_trackbar = ['desviacionSTD', 'tipo dato','ksize', 'umbral_min', 'umbral_max', 'L2Grad',]
    # parametros_trackbar = [[0,100], [1,2], [1,31], [0,255], [0,255], [0,1]]


"""Algunas preguntas de guia: 
¿En que zonas (debido a que) funciona mejor y en cuales no?, 
¿Que sucede con el ruido?, ¿Con que tipo de imagenes sacaría mejor provecho de los metodos?,
 ¿Que tipo de preprocesamientos, de los que ya conoce, se le ocurren que serıan utiles?, etc."""


"""6. [Opcional]: implemente una funcion que le permita realizar la conexion local de bordes, 
            a partir de las condiciones:"""