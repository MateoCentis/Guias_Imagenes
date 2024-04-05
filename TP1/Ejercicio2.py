import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
ruta = "imagenes_varias/micky.jpg"

imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
def click(event, x, y, flags, param):
    global imagen
    #Si se presiona el click izquierdo
    if event == cv.EVENT_LBUTTONDOWN:
        print("Valor de intensidad: ", imagen[x,y]) #Preguntar por que se van afuera los valores (x,y)
        # cv.imshow(str_win, imagen)

#1. Informe los valores de intensidad de puntos particulares de la imagen 
    #(opcional: determine la posici´on en base al click del mouse).
str_win = "DibujaWin"    
cv.namedWindow(str_win)
cv.setMouseCallback(str_win,click)
cv.imshow(str_win,imagen)
cv.waitKey(0)
cv.destroyAllWindows()
#2. Obtenga y grafique los valores de intensidad (perfil de intensidad) sobre una determinada fila o columna.
filaN = 250
intensidadFilaN = imagen[filaN,:]
plt.plot(intensidadFilaN)
plt.title(f"Perfil de intensidad fila {filaN}")
plt.show()

#3. Grafique el perfil de intensidad para un segmento de interés cualquiera.
segmento_interes = imagen[filaN,250:350]
indices = list(range(250,len(segmento_interes)+250)) #start,start+len
plt.plot(indices,segmento_interes,marker='o',linestyle='-')
plt.xlabel('Posición a lo largo del segmento')
plt.ylabel('Intensidad')
plt.title('Perfil de intensidad del segmento de interés')

plt.grid(True)
plt.show()