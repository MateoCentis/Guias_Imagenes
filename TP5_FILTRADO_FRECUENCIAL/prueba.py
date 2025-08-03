import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from utils import filtro_butterworth_pasa_bajos, filtro_butterworth_pasa_altos

def ecualizar_imagen(imagen):
    hist, bins = np.histogram(imagen.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imagen_ecualizada = cdf[imagen.astype('uint8')]
    return imagen_ecualizada

def filtro_homomorfico(imagen, gL, gH, D0, orden=5):# gL: ganancia de baja frecuencia
    imagen_float = np.float32(imagen)               # gH: ganancia de alta frecuencia
#                                                   # D0: frecuencia de corte
    imagen_log = np.log1p(imagen_float)             # orden: orden del filtro                              
    
    imagen_TDF = fft.fftshift(fft.fft2(imagen_log))    

    pasa_bajos = filtro_butterworth_pasa_bajos(imagen.shape, D0, orden)
    pasa_altos = filtro_butterworth_pasa_altos(imagen.shape, D0, orden)

    imagen_pasa_altos_TDF = imagen_TDF * pasa_altos
    imagen_pasa_bajos_TDF = imagen_TDF * pasa_bajos
    
    imagen_pasa_altos = fft.ifft2(fft.ifftshift(imagen_pasa_altos_TDF))
    imagen_pasa_bajos = fft.ifft2(fft.ifftshift(imagen_pasa_bajos_TDF))
    
    imagen_filtrada = gH * np.exp(imagen_pasa_altos.real) - gL * np.exp(imagen_pasa_bajos.real)
    
    # Clip o normalize?
    # np.uint8(np.clip(imagen_filtrada, 0, 255))
    # imagen_final = cv.normalize(imagen_filtrada, None, 0, 255, cv.NORM_MINMAX)
    
    return imagen_filtrada

def update_filtro_homomorfico(event):
    global gL, gH, D0, orden, casilla
    # Aplicar el filtro homomórfico
    casilla_filtrada = filtro_homomorfico(casilla, gL, gH, D0, orden)
    # Actualizar el gráfico con la imagen filtrada
    ax2.imshow(casilla_filtrada, cmap='gray')
    fig.canvas.draw_idle()

def update_gL(val):
    global gL
    gL = val 
    update_filtro_homomorfico(None)

def update_gH(val):
    global gH
    gH = val 
    update_filtro_homomorfico(None)

def update_D0(val):
    global D0
    D0 = val 
    update_filtro_homomorfico(None)

def update_orden(val):
    global orden
    orden = val
    update_filtro_homomorfico(None)

# Supongamos que ya tienes definida la función filtro_homomorfico
casilla = cv2.imread('Imagenes_Ej/casilla.tif', cv2.IMREAD_GRAYSCALE)
mostrar_casilla = False
if mostrar_casilla:
    # Supongamos que 'casilla' es tu imagen original

    # Parámetros iniciales para el filtro homomórfico
    gL = 1
    gH = 1
    D0 = 0.01
    orden = 3

    casilla_filtrada = filtro_homomorfico(casilla, gL, gH, D0, orden)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(casilla, cmap='gray')
    ax1.set_title('Imagen Original')

    ax2.imshow(casilla_filtrada, cmap='gray')
    ax2.set_title('Imagen Filtrada')

    plt.subplots_adjust(bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    ax_gL = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_gH = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_D0 = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_orden = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)

    s_gL = plt.Slider(ax_gL, 'gL', 0.1, 10.0, valinit=gL, valstep=0.1)
    s_gH = plt.Slider(ax_gH, 'gH', 0.1, 10.0, valinit=gH, valstep=0.1)
    s_D0 = plt.Slider(ax_D0, 'D0', 0.01, 2, valinit=D0)
    s_orden = plt.Slider(ax_orden, 'Orden', 1, 10, valinit=orden, valstep=1)

    # Definir la función de actualización para las trackbars
    s_gL.on_changed(update_gL)
    s_gH.on_changed(update_gH)
    s_D0.on_changed(update_D0)
    s_orden.on_changed(update_orden)

    plt.show()

reunion = cv2.imread('Imagenes_Ej/reunion.tif', cv2.IMREAD_GRAYSCALE)
mostrar_reunion = False
if mostrar_reunion:
    def update_filtro_homomorfico(event):
        global gL, gH, D0, orden, reunion
        reunion_filtrada = filtro_homomorfico(reunion, gL, gH, D0, orden)
        ax2.imshow(reunion_filtrada, cmap='gray')
        fig.canvas.draw_idle()


    gL = 1
    gH = 1
    D0 = 0.01  # Inicializar en 0.01 en lugar de 1
    orden = 3

    reunion_filtrada = filtro_homomorfico(reunion, gL, gH, D0, orden)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(reunion, cmap='gray')
    ax1.set_title('Imagen Original')

    ax2.imshow(reunion_filtrada, cmap='gray')
    ax2.set_title('Imagen Filtrada')

    plt.subplots_adjust(bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    ax_gL = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_gH = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_D0 = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_orden = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)

    s_gL = plt.Slider(ax_gL, 'gL', 0.1, 10.0, valinit=gL, valstep=0.1)
    s_gH = plt.Slider(ax_gH, 'gH', 0.1, 10.0, valinit=gH, valstep=0.1)
    s_D0 = plt.Slider(ax_D0, 'D0', 0.01, 2.0, valinit=D0)  # Cambiar los límites de 0.01 a 2.0
    s_orden = plt.Slider(ax_orden, 'Orden', 1, 10, valinit=orden, valstep=1)

    s_gL.on_changed(update_gL)
    s_gH.on_changed(update_gH)
    s_D0.on_changed(update_D0)
    s_orden.on_changed(update_orden)

    plt.show()


casilla_filtrada = filtro_homomorfico(casilla,8,7,0.8,7)

casilla_ecualizada = cv2.equalizeHist(casilla_filtrada)

reunion_filtrada = filtro_homomorfico(reunion,8,7,0.8,7)

reunion_ecualizada = cv2.equalizeHist(reunion_filtrada)

fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(casilla, cmap='gray')
axes[0, 0].set_title('Imagen Original - Casilla')

axes[0, 1].imshow(casilla_ecualizada, cmap='gray')
axes[0, 1].set_title('Imagen Ecualizada - Casilla')

axes[1, 0].imshow(reunion, cmap='gray')
axes[1, 0].set_title('Imagen Original - Reunion')

axes[1, 1].imshow(reunion_ecualizada, cmap='gray')
axes[1, 1].set_title('Imagen Ecualizada - Reunion')

plt.tight_layout()
plt.show()
#7, 0.8, 8, 7