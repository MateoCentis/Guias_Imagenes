import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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

def filtro_butterworth_pasa_bajos(size, D0, n):
    filas, columnas = size
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

##############################################################################################################
#                                                  Parte 1
##############################################################################################################
ruta = 'Imagenes_Ej/flores02.jpg'
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
size = imagen.shape

ejercicio1 = False
if ejercicio1:
    radio = 30

    filtro = filtro_pasa_bajos_ideal(size, radio)
    #Ver filtro
    plt.figure()
    plt.imshow(filtro, cmap='gray')
    plt.show()

    #normalizo para que no se me vayan los valores y aplico fftshift para que quede centrada
    imagen_filtrada = cv.normalize(aplicar_filtro(imagen, filtro),None,0,255,cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Bajos Ideal (Radio {})'.format(radio))
    plt.axis('off')
    plt.show()

##############################################################################################################
#                                                  Parte 2
##############################################################################################################
ejercicio2 = True
if ejercicio2:
    D0 = 40
    n = 5
    filtro = filtro_butterworth_pasa_bajos(size, D0, n)

    plt.figure()
    plt.imshow(filtro, cmap='gray')
    plt.show()

    imagen_filtrada = cv.normalize(aplicar_filtro(imagen, filtro),None,0,255,cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Bajos Butterworth')
    plt.axis('off')
    plt.show()

##############################################################################################################
#                                                  Parte 3
##############################################################################################################
ejercicio3 = False
if ejercicio3:
    size = 35
    sigma = 5
    H_gaussiano = filtro_gaussiano_pasa_bajos(size,sigma) # este ya es H ?
    # H_gaussiano = np.fft.fft2(h_gaussiano)
    magnitud = np.abs(H_gaussiano)
    fase = np.angle(H_gaussiano)

    # Se tiene que hacer separado porque por si sola tiene valores complejos
    H_magnitud_resize = cv.resize(magnitud,(imagen.shape[-1],imagen.shape[0]))

    fase_resize = cv.resize(fase, (imagen.shape[1], imagen.shape[0]))
    fase_extendida = np.tile(fase_resize, (imagen.shape[0] // size + 1, imagen.shape[1] // size + 1))[:imagen.shape[0], :imagen.shape[1]]
    
    H_gaussiano_resize = H_magnitud_resize*np.exp(1j*fase_extendida)
    imagen_filtrada =  cv.normalize(aplicar_filtro(imagen, H_gaussiano_resize),None,0,255,cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Bajos Gaussiano')
    plt.axis('off')
    plt.show()

#######################################################################################################################################################
#                                                       FILTROS PASA ALTOS
#######################################################################################################################################################
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
            distancia = np.sqrt((i - centro_fila)**2 + (j - centro_columna)**2)
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

# ruta = 'Imagenes_Ej/flores02.jpg'
# imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
# size = imagen.shape
# size = (size,size)
# Parte 1: Filtro pasa-altos ideal
ejercicio1 = True
if ejercicio1:
    radio = 30
    filtro = filtro_pasa_altos_ideal(size, radio)
    imagen_filtrada = cv.normalize(aplicar_filtro(imagen, filtro), None, 0, 255, cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Altos Ideal (Radio {})'.format(radio))
    plt.axis('off')
    plt.show()

# Parte 2: Filtro pasa-altos Butterworth
ejercicio2 = True
if ejercicio2:
    D0 = 40
    n = 5
    filtro = filtro_butterworth_pasa_altos(size, D0, n)
    imagen_filtrada = cv.normalize(aplicar_filtro(imagen, filtro), None, 0, 255, cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Altos Butterworth')
    plt.axis('off')
    plt.show()

# Parte 3: Filtro pasa-altos Gaussiano
ejercicio3 = True
if ejercicio3:
    size = 35
    sigma = 5
    H_gaussiano = filtro_gaussiano_pasa_altos(size, sigma)
    magnitud = np.abs(H_gaussiano)
    fase = np.angle(H_gaussiano)

    H_magnitud_resize = cv.resize(magnitud, (imagen.shape[1], imagen.shape[0]))

    fase_resize = cv.resize(fase, (imagen.shape[1], imagen.shape[0]))
    fase_extendida = np.tile(fase_resize, (imagen.shape[0] // size + 1, imagen.shape[1] // size + 1))[:imagen.shape[0], :imagen.shape[1]]

    H_gaussiano_resize = H_magnitud_resize * np.exp(1j * fase_extendida)

    imagen_filtrada = cv.normalize(aplicar_filtro(imagen, H_gaussiano_resize), None, 0, 255, cv.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('Imagen Filtrada con Pasa-Altos Gaussiano')
    plt.axis('off')
    plt.show()
