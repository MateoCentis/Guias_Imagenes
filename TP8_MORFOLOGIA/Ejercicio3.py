import cv2 as cv
import numpy as np
from utils import ruido_sal_y_pimienta
imagen = cv.imread("Morfologia_Im/fosforos.jpg")
imagen2 = cv.imread("Morfologia_Im/createch01.png")

imagen_gris = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
imagen_gris2 = cv.cvtColor(imagen2,cv.COLOR_BGR2GRAY)

_,imagen_bin = cv.threshold(imagen_gris, 240,255,cv.THRESH_BINARY)
_,imagen_bin2 = cv.threshold(imagen_gris2, 127,255,cv.THRESH_BINARY)

comparacion = False
if comparacion: 
    ksize = 3
    kernel = np.ones((ksize,ksize), np.uint8)

    imagen_erosion = cv.erode(imagen_bin,kernel,iterations = 1)

    imagen_erosion2 = cv.erode(imagen_bin2,kernel,iterations = 1)

    morfologia_bordes1 = imagen_bin - imagen_erosion
    morfologia_bordes2 = imagen_bin2 - imagen_erosion2

    canny1 = cv.Canny(imagen_bin, threshold1=100, threshold2=200, L2gradient=0)
    sobel1 = cv.Sobel(imagen_bin, ddepth=cv.CV_8U, dx=1, dy=1, ksize=ksize)

    canny2 = cv.Canny(imagen_bin2, threshold1=100, threshold2=200, L2gradient=0)
    sobel2 = cv.Sobel(imagen_bin2, ddepth=cv.CV_8U, dx=2, dy=1, ksize=ksize)

    cv.imshow("Morfologia 1", morfologia_bordes1)
    cv.imshow("Morfologia 2", morfologia_bordes2)

    cv.imshow("Canny 1", canny1)
    cv.imshow("Canny 2", canny2)

    cv.imshow("Sobel 1", sobel1)
    cv.imshow("Sobel 2", sobel2)

    cv.waitKey(0)

#---------------------------------------------Agregarle ruido sal y pimienta-------------------------------------------------
comparacion_ruido = True
if comparacion_ruido:
    imagen_gris_ruidosa = ruido_sal_y_pimienta(imagen_gris, 11000)
    imagen_gris_ruidosa2 = ruido_sal_y_pimienta(imagen_gris2, 11000)

    _,imagen_bin_ruidosa = cv.threshold(imagen_gris_ruidosa, 240,255,cv.THRESH_BINARY)
    _,imagen_bin_ruidosa2 = cv.threshold(imagen_gris_ruidosa2, 127,255,cv.THRESH_BINARY)

    #Volver a aplicar procesamiento: se ve que es menos susceptible al ruido, de todas formas habría que sacar
        #el ruido antes de aplicar sobel o canny (nunca te mandas derecho)

    ksize = 3
    kernel = np.ones((ksize,ksize), np.uint8)

    imagen_erosion = cv.erode(imagen_bin,kernel,iterations = 3)

    imagen_erosion2 = cv.erode(imagen_bin2,kernel,iterations = 3)

    morfologia_bordes1 = imagen_bin_ruidosa - imagen_erosion
    morfologia_bordes2 = imagen_bin_ruidosa2 - imagen_erosion2

    canny1 = cv.Canny(imagen_bin_ruidosa, threshold1=100, threshold2=200, L2gradient=0)
    sobel1 = cv.Sobel(imagen_bin_ruidosa, ddepth=cv.CV_8U, dx=1, dy=1, ksize=ksize)

    canny2 = cv.Canny(imagen_bin_ruidosa2, threshold1=100, threshold2=200, L2gradient=0)
    sobel2 = cv.Sobel(imagen_bin_ruidosa2, ddepth=cv.CV_8U, dx=2, dy=1, ksize=ksize)

    cv.imshow("Morfologia 1", morfologia_bordes1)
    cv.imshow("Morfologia 2", morfologia_bordes2)

    cv.imshow("Canny 1", canny1)
    cv.imshow("Canny 2", canny2)

    cv.imshow("Sobel 1", sobel1)
    cv.imshow("Sobel 2", sobel2)

    cv.waitKey(0)