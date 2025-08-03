import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import mse

def ver_imagen_en_frecuencia(imagen):
    imagen_fft = np.fft.fftshift(np.fft.fft2(imagen))
    imagen_fft = np.log(np.abs(imagen_fft))
    plt.imshow(imagen_fft, cmap='gray')
    plt.show()

#---------------------------------------------Localización de picos-------------------------------------------------

imagen_ruido = cv.imread("Imagenes_Ej/img_degradada.tif",cv.IMREAD_GRAYSCALE)

ver_graficos = False
if ver_graficos:
    plt.imshow(imagen_ruido, cmap='gray')
    plt.show()
    ver_imagen_en_frecuencia(imagen_ruido)
    plt.hist(imagen_ruido.ravel(), bins=256)
    plt.show()

#---------------------------------------------Implementación de filtros-------------------------------------------------


def rechazabanda_ideal(shape, centro, ancho):
    x = np.arange(shape[0])
    y = np.arange(shape[1])

    xx, yy = np.meshgrid(x,y)

    distancias = np.sqrt((xx - centro[0])**2 + (yy - centro[1])**2)
    filtro = np.where((distancias > (ancho/2)), 1, 0)
    
    return filtro

def rechazabanda_butterworth(shape : tuple[int, int], centro : tuple[int, int], ancho : int, orden : int):
    x = np.arange(shape[0])
    y = np.arange(shape[1])

    xx, yy = np.meshgrid(x,y)

    distancias = np.sqrt((xx - centro[0])**2 + (yy - centro[1])**2)
    filtro = 1 - 1 / (1 + (distancias / ancho)**(2 * orden))
    
    return filtro

def notch_ideal(shape : tuple[int, int], centro : tuple[int, int], ancho : int):
    x = np.arange(shape[0])
    y = np.arange(shape[1])

    xx, yy = np.meshgrid(x,y)

    distancias = np.sqrt((xx - centro[0])**2 + (yy - centro[1])**2)
    filtro = np.where((distancias > (ancho/2)) | (distancias > (shape[0] - ancho / 2)), 1, 0)
    
    return filtro

def aplicar_filtro(imagen, filtro):#Se asume al filtro ya definido en frecuencia
    imagen_fft = np.fft.fftshift(np.fft.fft2(imagen))

    imagen_filtrada_fft = imagen_fft * filtro
    
    imagen_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(imagen_filtrada_fft)))

    return imagen_filtrada


def graficar_imagenes(imagenes, titulos):
    num_imagenes = len(imagenes)
    filas = math.ceil(num_imagenes / 2)
    fig, axs = plt.subplots(filas, 2, figsize=(15, 8))
    for i in range(num_imagenes):
        fila = i // 2
        columna = i % 2
        axs[fila, columna].imshow(imagenes[i], cmap='gray')
        axs[fila, columna].axis('off')
        axs[fila, columna].set_title(titulos[i])
    plt.tight_layout()
    plt.show()

def notch_ideal(shape: tuple[int, int], radio: int, u_k: int, v_k: int):# u_k y v_k los valores de frecuencia
    (M, N) = shape

    H_0_u = np.repeat(np.arange(M), N).reshape((M, N))
    H_0_v = np.repeat(np.arange(N), M).reshape((N, M)).transpose()

    D_uv = np.sqrt((H_0_u - M / 2 + u_k) ** 2 + (H_0_v - N / 2 + v_k) ** 2)
    D_muv = np.sqrt((H_0_u - M / 2 - u_k) ** 2 + (H_0_v - N / 2 - v_k) ** 2)

    selector_1 = D_uv <= radio
    selector_2 = D_muv <= radio

    selector = np.logical_or(selector_1, selector_2)

    H = np.ones((M, N))
    H[selector] = 0

    return H

def notch_butterworth(shape: tuple[int, int], radio: int, u_k: int, v_k: int, orden: int):
    (M, N) = shape

    H_0_u = np.repeat(np.arange(M), N).reshape((M, N))
    H_0_v = np.repeat(np.arange(N), M).reshape((N, M)).transpose()

    D_uv = np.sqrt((H_0_u - M / 2 + u_k) ** 2 + (H_0_v - N / 2 + v_k) ** 2)
    D_muv = np.sqrt((H_0_u - M / 2 - u_k) ** 2 + (H_0_v - N / 2 - v_k) ** 2)

    # Calculando las funciones de transferencia de Butterworth
    H_uv = 1 - 1 / (1 + (D_uv / radio) ** (2 * orden))
    H_muv = 1 - 1 / (1 + (D_muv / radio) ** (2 * orden))

    H = H_uv * H_muv

    return H

#---------------------------------------------Aplique los filtros-------------------------------------------------

imagen = cv.imread("Imagenes_Ej/img.tif", cv.IMREAD_GRAYSCALE)
shape = imagen.shape

ancho = 40
orden = 9

H1 = rechazabanda_ideal(shape, (197.5,218), ancho)
H2 = rechazabanda_ideal(shape, (57,218), ancho)
filtro_rechazabanda_ideal = H1*H2
imagen_rechazabanda_ideal = aplicar_filtro(imagen_ruido,filtro_rechazabanda_ideal)

H1 = rechazabanda_butterworth(shape, (197.5,218), ancho, orden)
H2 = rechazabanda_butterworth(shape, (57,218), ancho, orden)
filtro_rechazabanda_butterworth = H1*H2
imagen_rechazabanda_butterworth = aplicar_filtro(imagen_ruido,filtro_rechazabanda_butterworth)

# notch_ideal
radio = 50
H1 = notch_ideal(shape, radio, 51, 70) #rango de valores, se obtiene al observar el histograma 
H2 = notch_ideal(shape, radio, 122, 132)
H3 = notch_ideal(shape, radio, 170, 184)

filtro_notch_ideal = H1*H2*H3
imagen_notch_ideal = aplicar_filtro(imagen_ruido,filtro_notch_ideal)

# notch_butterworth
H1 = notch_butterworth(shape, radio, 51, 70, orden)
H2 = notch_butterworth(shape, radio, 122, 132, orden)
H3 = notch_butterworth(shape, radio, 170, 184, orden)

filtro_notch_butterworth = H1*H2*H3
imagen_notch_butterworth = aplicar_filtro(imagen_ruido,filtro_notch_butterworth)


ver_filtros = False
if ver_filtros:
    ver_filtro_rechazabanda_ideal = np.abs(filtro_rechazabanda_ideal)
    ver_filtro_notch_ideal = np.abs(filtro_notch_ideal)
    ver_filtro_rechazabanda_butterworth = np.abs(filtro_rechazabanda_butterworth)
    ver_filtro_notch_butterworth = np.abs(filtro_notch_butterworth)
    titulos = ['Filtro Rechazabanda Ideal', 'Filtro Notch Ideal', 'Filtro Rechazabanda Butterworth', 'Filtro Notch Butterworth']
    imagenes = [ver_filtro_rechazabanda_ideal, ver_filtro_notch_ideal, ver_filtro_rechazabanda_butterworth, ver_filtro_notch_butterworth]
    graficar_imagenes(imagenes,titulos)

ver_graficos2 = False
if ver_graficos2:
    titulos = ['Imagen Original', 'Imagen con Ruido', 'Imagen Filtrada Rechazabanda Ideal',
               'Imagen Filtrada Notch Ideal', 'Imagen Filtrada Rechazabanda Butterworth',
               'Imagen Filtrada Notch Butterworth']
    
    imagenes = [imagen, imagen_ruido, imagen_rechazabanda_ideal,
                imagen_notch_ideal, imagen_rechazabanda_butterworth,
                imagen_notch_butterworth]
    
    graficar_imagenes(imagenes,titulos)

#---------------------------------------------Cálculo del MSE-------------------------------------------------
titulos = ['MSE Rechazabanda Ideal','MSE Rechazabanda Butterworth',
            'MSE Notch Ideal','MSE Notch Butterworth']
imagenes = [imagen_rechazabanda_ideal,imagen_rechazabanda_butterworth,
            imagen_notch_ideal, imagen_notch_butterworth]

print("MSE imagen ruido: ", mse(imagen,imagen_ruido))
for i in range(len(titulos)):
    error = mse(imagen, imagenes[i])
    print(titulos[i],": ", error)

#---------------------------------------------Imagen solo ruido-------------------------------------------------
mostrar_imagen_solo_ruido = False
if mostrar_imagen_solo_ruido:
    H1 = notch_butterworth(shape, radio, 51, 70, orden)
    H2 = notch_butterworth(shape, radio, 122, 132, orden)
    H3 = notch_butterworth(shape, radio, 170, 184, orden)

    filtro_notch_butterworth = H1*H2*H3
    imagen_solo_ruido = aplicar_filtro(imagen_ruido,1-filtro_notch_butterworth)
    plt.imshow(imagen_solo_ruido)
    plt.show()
#----------------------------------Aplicación de filtros a otras imágenes-------------------------------------------------
parte6 = False
if parte6:#Se deja porque alta paja
    luna_ruido = cv.imread("Imagenes_Ej/noisy_moon.jpg", cv.IMREAD_GRAYSCALE)
    shape_luna = luna_ruido.shape
    head_ruido = cv.imread("Imagenes_Ej/HeadCT_degradada.jpg", cv.IMREAD_GRAYSCALE)
    shape_head = head_ruido.shape
    