import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
imagen = cv.imread("Morfologia_Im/createch01.png")
# Implemente una secuencia de operaciones con diferentes EE's para obtener el logo de la imagen
    #PISTA: extraer el logo por partes y combinarlas
imagen_gris = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)

_,imagen_bin = cv.threshold(imagen_gris, 127,255,cv.THRESH_BINARY)
imagen_bin = imagen_bin - 255

#50,50
#30,30
kernel = np.ones((10,10),np.uint8)

imagen_erosion = cv.erode(imagen_bin,kernel,iterations = 1)

plt.imshow(imagen_erosion, cmap='gray')
plt.show()
# cv.imshow("e", imagen_erosion)
# cv.waitKey(0)