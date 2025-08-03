import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
imagen = cv.imread("Morfologia_Im/notas01.png")

imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

_,imagen_bin = cv.threshold(imagen_gris, 210, 255, cv.THRESH_BINARY)
imagen_bin = imagen_bin - 255
#---------------------------------------Obtener solo lo que toca las lineas-------------------------------------------------
solo_lineas = False
if solo_lineas:
    #¿como eliminarıa todo lo que no esta en contacto con las líneas del pentagrama?
    # kernel = np.zeros((10,10), dtype=np.uint8)
    kernel = np.zeros((3,3), dtype=np.uint8)

    # kernel[2,:] = 1
    kernel[1,:] = 1
    # kernel[:,1] = 1

    res = cv.morphologyEx(imagen_bin, cv.MORPH_HITMISS,kernel,iterations=80)
    # res = cv.erode(imagen_bin,kernel,iterations=2)
    plt.imshow(res, cmap='gray')
    plt.show()
    semilla = res.copy()
    mascara = imagen_bin
    reconstruido = morphology.reconstruction(semilla,mascara,method='dilation')

    respuesta = reconstruido 
    plt.imshow(respuesta, cmap='gray')

    plt.show()
#---------------------------------------------Separar notas musicales-------------------------------------------------
notas = True
if notas:
    # kernel = np.zeros((10,10), dtype=np.uint8)
    kernel = np.zeros((3,3), dtype=np.uint8)

    # kernel[2,:] = 1
    kernel[:,1] = 1
    # kernel[:,1] = 1

    res = cv.morphologyEx(imagen_bin, cv.MORPH_HITMISS,kernel,iterations=1)
    # res = cv.erode(imagen_bin,kernel,iterations=2)
    
    
    # semilla = res.copy()
    # mascara = imagen_bin
    # reconstruido = morphology.reconstruction(semilla,mascara,method='dilation')

    plt.imshow(res, cmap='gray')

    plt.show()
