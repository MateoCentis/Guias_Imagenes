import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def reconstruir_magnitud_fase(magnitud, fase):
    # Reconstruir en frecuencia
    fft_reconstruida = magnitud * np.exp(1j * fase)
    # Reconstruir en dominio espacial con la TFI
    imagen_reconstruida = np.fft.ifft2(fft_reconstruida).real
    # Normalizar al rango 0-255 (SI NO NO SE VE)
    imagen_reconstruida = np.uint8(cv.normalize(imagen_reconstruida, None, 0, 255, cv.NORM_MINMAX))
    return imagen_reconstruida

ejercicio1 = False
if ejercicio1:
    ruta = "Imagenes_Ej/chairs.jpg"

    imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)

    plt.imshow(imagen, cmap='gray')
    plt.title("Imagen original")
    plt.axis('off')
    plt.show()

    fft_imagen = np.fft.fft2(imagen)

    # Magnitud y fase
    magnitud = np.abs(fft_imagen)
    fase = np.angle(fft_imagen)

    # Normalizar magnitud y fase
    magnitud_normalizada = np.log1p(magnitud)

    #Reconstrucción
    imagen_magnitud = reconstruir_magnitud_fase(magnitud_normalizada, np.zeros_like(fase))*30 #La imagen de magnitud se ve muy oscura
    imagen_fase = reconstruir_magnitud_fase(np.ones_like(magnitud), fase)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(imagen_magnitud, cmap='gray')
    axs[0].set_title("Imagen reconstruida solo con magnitud")
    axs[0].axis('off')
    axs[1].imshow(imagen_fase, cmap='gray')
    axs[1].set_title("Imagen reconstruida solo con fase")
    axs[1].axis('off')
    plt.show()

################################################################################################################
#                                                     Ejercicio 2
################################################################################################################

ruta1 = "ImgsPracticos/puente.jpg"
ruta2 = "ImgsPracticos/ferrari-c.png"

puente = cv.imread(ruta1, cv.IMREAD_GRAYSCALE)

ferrari = cv.imread(ruta2, cv.IMREAD_GRAYSCALE)

fft_puente = np.fft.fft2(puente)
fft_ferrari = np.fft.fft2(ferrari)

magnitud_puente = np.abs(fft_puente)
fase_puente = np.angle(fft_puente)

magnitud_ferrari = np.abs(fft_ferrari)
fase_ferrari = np.angle(fft_ferrari)

puente_con_fase_ferrari = reconstruir_magnitud_fase(magnitud_puente,fase_ferrari)
ferrari_con_fase_puente = reconstruir_magnitud_fase(magnitud_ferrari,fase_puente)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(puente, cmap='gray')
axs[0, 0].set_title('Puente (f)')

axs[0, 1].imshow(ferrari, cmap='gray')
axs[0, 1].set_title('Ferrari (g)')

axs[1, 0].imshow(puente_con_fase_ferrari, cmap='gray')
axs[1, 0].set_title('Puente con fase ferrari')

axs[1, 1].imshow(ferrari_con_fase_puente, cmap='gray')
axs[1, 1].set_title('Ferrari con fase puente')

plt.tight_layout()
plt.show()