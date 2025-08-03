import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from icecream import ic
from utils import graficar_histogramas_subplots

def ruido_gausiano(shape, media=0, desviacion_estandar=10):
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

def mostrar_imagen(imagen):
    plt.imshow(imagen, cmap='gray')
    plt.show()

def generar_imagen_franjas(height, width):
    image = np.zeros((height, width), dtype=np.uint8)
    third_width = width // 3
    image[:, :third_width] = 180  # Franja clara
    image[:, third_width:2*third_width] = 120  # Franja media
    image[:, 2*third_width:] = 60  # Franja oscura
    return image

def graficar_histograma(imagen, bins=256):
    hist = cv.calcHist([imagen], [0], None, [bins], [0, 256])
    
    # Graficar el histograma
    plt.plot(hist, color='black')
    plt.title('Histograma de la imagen')
    plt.xlabel('Valor de píxel')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

#---------------------------------------------Ejercicio 1-------------------------------------------------
ejercicio1 = False
if ejercicio1:
    imagen = cv.imread("Imagenes_Ej/chairs.jpg",cv.IMREAD_GRAYSCALE)
    shape_imagen = imagen.shape
    ruido_gaussiano = ruido_gausiano(shape_imagen)
    ic(np.mean(ruido_gaussiano))
    imagen_ruido_gaussiano = imagen + ruido_gaussiano

    ruido_uniforme = ruido_uniforme(shape_imagen)
    ic(np.mean(ruido_uniforme))
    imagen_ruido_uniforme = imagen + ruido_uniforme

    imagen_sal_pimienta = ruido_sal_y_pimienta(imagen,200000)
    plt.imshow(imagen_sal_pimienta, cmap='gray')
    plt.show()
    ruido_impulsivo_unimodal = ruido_impulsivo_unimodal(shape_imagen)
    imagen_ruido_impulsivo_unimodal = imagen + ruido_impulsivo_unimodal

    ruido_exponencial = ruido_exponencial(shape_imagen)
    ic(np.mean(ruido_exponencial))
    imagen_ruido_exponencial = imagen + ruido_exponencial
    
    imagenes = [imagen, imagen_ruido_gaussiano, imagen_ruido_uniforme, imagen_sal_pimienta, imagen_ruido_impulsivo_unimodal, imagen_ruido_exponencial]
    titulos = ['Original', 'Gaussian Noise', 'Uniform Noise', 'Salt & Pepper Noise', 'Impulsive Unimodal Noise', 'Exponential Noise']

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imagenes[i], cmap='gray')
        ax.set_title(titulos[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#---------------------------------------------Ejercicio 2-------------------------------------------------
alto, ancho = 600, 600

imagen_franjas = generar_imagen_franjas(alto, ancho)
franjas_shape = imagen_franjas.shape

media = 0
desvio = 10

bajo = -20
alto = 20

escala = 10

ruido_gausiano = ruido_gausiano(franjas_shape,media,desvio)
ruido_uniform = ruido_uniforme(franjas_shape,bajo,alto)
ruido_exponencial = ruido_exponencial(franjas_shape,escala)

imagen_franjas_ruido = imagen_franjas + ruido_gausiano + ruido_uniform + ruido_exponencial

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].imshow(imagen_franjas, cmap='gray')
axs[0, 0].set_title('Imagen Original')
axs[0, 1].hist(imagen_franjas.ravel(), bins=256, color='black')
axs[0, 1].set_title('Histograma Original')

axs[1, 0].imshow(imagen_franjas_ruido, cmap='gray')
axs[1, 0].set_title('Imagen con Ruido')
axs[1, 1].hist(imagen_franjas_ruido.ravel(), bins=256, color='black')
axs[1, 1].set_title('Histograma con Ruido')

plt.tight_layout()
plt.show()