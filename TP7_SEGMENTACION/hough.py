import cv2 as cv
import numpy as np
from utils import ventana_trackbars, trackbar_transformacion
import math
def dibujar_lineas(imagen, lines, probabilistica):
    if lines is not None:
        if probabilistica == 0:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(imagen, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        else: #probabilistica 1
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(imagen, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    return imagen
    
def dibujar_imagenes_Hough(imagen_gris, umbral1=50, umbral2=200, sizeApertura=3, #Canny
                           rho=1, theta=np.pi/180, umbral_linea=150,#Hough
                           probabilistica=0, minLongLinea=50, maxDistEntreLineas=10):#si se usa houghLinesP o houghLines
   # 1. Umbralizar la imagen 
    imagen_bordes = cv.Canny(imagen_gris, umbral1, umbral2,None,sizeApertura)
   
   # 2. copiar imagen a color para hacer líneas encima
    copia_bordes = cv.cvtColor(imagen_bordes, cv.COLOR_GRAY2BGR)
   
   # 3. Obtener líneas con la transformada de Hough
    if probabilistica == 0:
       srn = 0
       stn = 0
       lines = cv.HoughLines(imagen_bordes, rho, theta, umbral_linea, None, srn, stn)
    else:
       lines = cv.HoughLinesP(imagen_bordes, rho, theta, umbral_linea, None, minLongLinea, maxDistEntreLineas)
    # 4. Dibujar lineas
    dibujar_lineas(copia_bordes, lines, probabilistica)

    return copia_bordes


ruta = "Imagenes_Ej/chairs.jpg"
imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)

variables_trackbar = ['umbral1', 'umbral2', 'sizeApertura', 
                      'rho', 'theta', 'umbral_linea', 
                      'probabilistica', 'minLongLinea', 'maxDistEntreLines']
parametros_trackbar = [[50,255],[200,255],[3,10],
                       [1,100],[180,360],[150,255],
                       [0,1], [50,100],[10,100]]

def transformacion(imagen, valores_trackbar):
    umbral1 = valores_trackbar[0]
    umbral2 = valores_trackbar[1]
    sizeApertura = np.max([valores_trackbar[2],1]).astype(np.uint8)
    rho = valores_trackbar[3]
    theta = valores_trackbar[4]
    umbral_linea = valores_trackbar[5]
    probabilistica = valores_trackbar[6]
    minLongLinea = valores_trackbar[7]
    maxDistEntreLineas = valores_trackbar[8]
    imagen_salida = dibujar_imagenes_Hough(imagen, umbral1, umbral2, sizeApertura, rho, theta, umbral_linea, probabilistica, minLongLinea, maxDistEntreLineas)
    return imagen_salida

ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)
# imagen_salida = dibujar_imagenes_Hough(imagen,probabilistica=1)
# cv.imshow("LINEAS",imagen_salida)
# cv.waitKey(0)

##############################################################################################################
todo = False
if todo:
    hough_comun = False
    imagen_bordes = cv.Canny(imagen, 50, 200, None, 3)
    copia_bordes = cv.cvtColor(imagen_bordes, cv.COLOR_GRAY2BGR)
    copia_bordesP = np.copy(copia_bordes)
    if hough_comun:
    # #---------------------------------------------Hough común-------------------------------------------------
        lines = cv.HoughLines(imagen, 1, np.pi/180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(copia_bordes, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", copia_bordes)

    hough_p = False
    if hough_p:
    #-------------------------------------------Hough probabilística-------------------------------------------------
        linesP = cv.HoughLinesP(imagen_bordes, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(copia_bordesP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", copia_bordesP)
    
    cv.imshow("Source", imagen)
    cv.waitKey(0)
