import cv2 as cv
import numpy as np
import time
def encontrar_size_optimo(imagen):
    filas, columnas = imagen.shape[:2]

    filas_optimas = cv.getOptimalDFTSize(filas)
    columnas_optimas = cv.getOptimalDFTSize(columnas)

    return (filas_optimas, columnas_optimas)

def agregar_ceros(imagen, size_objetivo):

    imagen_con_ceros = np.zeros(size_objetivo, dtype=imagen.dtype)

    if imagen.shape[0] > size_objetivo[0]: # Si es más chica en filas
        imagen_con_ceros[:size_objetivo[0], :imagen.shape[1]] = imagen[:size_objetivo[0],:]
    elif imagen.shape[1] > size_objetivo[1]: #Si es más chica en columnas
        imagen_con_ceros[:imagen.shape[0], :size_objetivo[1]] = imagen[:, :size_objetivo[1]]
    else: #Si es más grande en ambas se hace full
        imagen_con_ceros[:imagen.shape[0], :imagen.shape[1]] = imagen

    return imagen_con_ceros

ruta = "imagenes_varias/img_alta_res.jpg"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

size_optimo = encontrar_size_optimo(imagen)
print("Tamaño original: ", imagen.shape)

print("Tamaño óptimo: ", size_optimo)

# Obtención de imágenes con ceros
imagen_con_ceros_optima = agregar_ceros(imagen, size_optimo) #(Nopt)x(Mopt)
imagen_con_ceros_menor = agregar_ceros(imagen, (size_optimo[0] - 1, size_optimo[1] - 1)) #(Nopt - 1)x(Mopt - 1)

# tiempo_inicio = time.time()

# imagen_FFT = np.fft.fft2(imagen)
# imagen_con_ceros_optima_FFT = np.fft.fft2(imagen_con_ceros_optima)
# imagen_con_ceros_menor_FFT = np.fft.fft2(imagen_con_ceros_menor)

# tiempo_fin = time.time()
# print("Tiempo de cálculo FFT:", tiempo_fin - tiempo_inicio)

tiempo_inicio = time.time()

imagen_DFT = cv.dft(np.float32(imagen), flags = cv.DFT_COMPLEX_OUTPUT) #0.22
tiempo_fin = time.time()
print("Tiempo de cálculo DFT imagen original:", tiempo_fin - tiempo_inicio)

tiempo_inicio = time.time()
imagen_con_ceros_optima_DFT = cv.dft(np.float32(imagen_con_ceros_optima), flags = cv.DFT_COMPLEX_OUTPUT) #0.11
tiempo_fin = time.time()

print("Tiempo de cálculo DFT imagen con ceros optima:", tiempo_fin - tiempo_inicio)

tiempo_inicio = time.time()
imagen_con_ceros_menor_DFT = cv.dft(np.float32(imagen_con_ceros_menor), flags = cv.DFT_COMPLEX_OUTPUT) #0.33

tiempo_fin = time.time()

print("Tiempo de cálculo DFT imagen con ceros menor:", tiempo_fin - tiempo_inicio)
