import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology, util

imagen = cv.imread("Morfologia_Im/Caracteres.jpeg")

imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
# imagen_gris = cv.resize(imagen_gris, (imagen_gris.shape[1]*2,imagen_gris.shape[0]*2)) #Duplico tamaño de imagen

_,imagen_bin = cv.threshold(imagen_gris, 231, 255, cv.THRESH_BINARY) #Umbralizado
imagen_bin = imagen_bin - 255
plt.imshow(imagen_bin, cmap='gray')
plt.show()

kernelI = np.zeros((18,10), dtype=np.uint8)
kernelI[1:16,2:6] = 1

ksize = 3
kernelT = np.zeros((20, 20), dtype=np.uint8)
kernelT[1:16,8:12] = 1
kernelT[1:4,5:16] = 1
plt.imshow(kernelT, cmap='gray')
plt.show()


kernelE = np.ones((ksize, ksize), dtype=np.uint8)
kernelE[1,1:2] = 0

resultadoI = cv.morphologyEx(imagen_bin,cv.MORPH_HITMISS,kernelI,iterations=1)
resultadoT = cv.morphologyEx(imagen_bin,cv.MORPH_HITMISS,kernelT,iterations=1)
# plt.imshow(resultadoI, cmap='gray')
# plt.show()
# resultadoE = cv.morphologyEx(imagen_bin,cv.MORPH_HITMISS,kernelE,iterations=2)
# resultadoI = cv.erode(imagen_bin,kernelI,iterations=1)
# resultadoT = cv.erode(imagen_bin,kernelT,iterations=1)
# resultadoE = cv.erode(imagen_bin,kernelE,iterations=1)

combinado_hit_or_miss = resultadoT

# plt.imshow(combinado_hit_or_miss, cmap='gray')
# plt.show()
# #Reconstrucción por dilatación
semilla = combinado_hit_or_miss.copy()
mascara = imagen_bin
reconstruido = morphology.reconstruction(semilla,mascara,method='dilation')
plt.imshow(reconstruido, cmap='gray')
plt.show()

