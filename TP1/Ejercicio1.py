import cv2 as cv
import numpy as np  
import argparse
import matplotlib.pyplot as plt

def mostrar_imagenes(rutas_imagenes):
    num_imagenes = len(rutas_imagenes)
    fig, axs = plt.subplots(1, num_imagenes, figsize=(15, 5))
    
    for i, ruta_imagen in enumerate(rutas_imagenes):
        imagen = cv.imread(ruta_imagen)
        imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)  # Convertir la imagen a RGB para Matplotlib
        axs[i].imshow(imagen_rgb)
        axs[i].axis('off')  # Ocultar ejes
        
    plt.show()

# 1. Leer varias imágenes
    #La png no anda
    #Chequear bien el path, ya que si estamos parados desde GUIAS_IMAGENES debemos poner TPX en la ruta
# ruta1 = "TP1/Fondos/anorLondo.jpg"
# ruta2 = "TP1/Fondos/Irithyl.jpg"
# ruta3 = "TP1/Fondos/mrFox.jpg"
ruta1 = "imagenes_varias/micky.jpg"
ruta2 = "imagenes_varias/sentarse.jpg"
ruta3 = "imagenes_varias/simpsons.jpg"
ruta4 = "imagenes_varias/pikachu.jpg"
imagen1 = cv.imread(ruta1)
imagen2 = cv.imread(ruta2)
imagen3 = cv.imread(ruta3)
imagen4 = cv.imread(ruta4)
rutas = [ruta1, ruta2, ruta3,ruta4]
imagenes = [imagen1, imagen2, imagen3,imagen4]

for imagen in imagenes:
    if imagen is not None:
        # 2. Mostrar información de las imágenes
        print("Dimensiones de la imagen: ", imagen.shape)
        print("Tipo de dato de la imagen", imagen.dtype)
        
        # 3. Investigar formatos de las imágenes (RGB y eso?) y leer y escribir un valor puntual de la imagen (da error ahora)
        # xr = 400
        # yr = 400
        # valor_pixel = imagen[yr,xr]   # Lectura
        # xw = 500
        # yw = 500
        # imagen[yr,xr] = valor_pixel # Escritura

        # 4. Hacer pasaje por parámetros 
        ap = argparse.ArgumentParser()
        ap.add_argument("-i","--imagen",required=False,help="Ruta de la imagen a cargar")
        args = vars(ap.parse_args())

        ruta_imagen = args["imagen"]     
        imagen_parse = cv.imread(ruta_imagen)

        if imagen_parse is None:
            print("No se cargó la imagen por parámetros")
        else:
            print("JOYA")
        # comando (anda): python TP1/Ejercicio1.py -i TP1/Fondos/mrFox.jpg
            
        # 5. Definir y recortar un ROI
        x0 = 200
        x1 = 400
        y0 = 200
        y1 = 400
        imagen_ROI = imagen[x0:x1,y0:y1].copy()
        cv.imshow("ROI",imagen_ROI)

        
        # 7. Dibuje sobre la imagen líneas, cíırculos y rectángulos 
        cv.line(imagen, (x0,y0), (x1,y0), (0,255,0), 2)
        cv.circle(imagen, (x0,y0), 10, (0,255,0), -1)
        cv.rectangle(imagen, (x0,y0), (x1,y1), (0,255,0), 2)
        
        # Mostrar la imagen en una ventana hasta que se presione una tecla y destruir ventana
        cv.imshow("Hello World", imagen)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No se pudo leer la imagen. Verifica la ruta del archivo y la integridad del archivo.")



# 6. Hacer una función que permita dibujar varias imágenes en una ventana (subplots?)
mostrar_imagenes(rutas)


#(opcional 7): defina la posición en base al click del mouse.

refPt = []
def click(event, x, y, flags, param):
    global refPt
    #Si se presiona el click izquierdo
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
    elif event == cv.EVENT_LBUTTONUP:
        refPt.append((x,y))
        #dibuja la linea entre los puntos
        cv.line(imagen, refPt[0], refPt[1], (0,255,0), 2)
        cv.imshow(str_win, imagen)

str_win = "DibujaWin"    
cv.namedWindow(str_win)
cv.setMouseCallback(str_win,click)

while True:
    cv.imshow(str_win, imagen)
    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        break

