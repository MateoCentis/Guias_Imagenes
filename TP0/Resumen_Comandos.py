import cv2
import numpy as np  
import argparse
# 1. LECTURA DE UNA IMAGEN
imagen = cv2.imread("imagenes_varias/micky.jpg")

# Verificar si la lectura de la imagen fue exitosa
if imagen is not None:
    # Mostrar dimensiones y tipo de dato de la imagen
    print("Dimensiones de la imagen: ", imagen.shape)
    print("Tipo de dato de la imagen", imagen.dtype)
    
    # Mostrar la imagen en una ventana hasta que se presione una tecla y destruir ventana
    cv2.imshow("Hello World", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se pudo leer la imagen. Verifica la ruta del archivo y la integridad del archivo.")

# 2. GENERAR UNA NUEVA IMAGEN DE ZEROS, CONVERTIR A ESCALA DE GRISES
# imagen_nueva = np.zeros(imagen.shape, dtype = imagen.dtype)
# imagen_nueva = np.zeros_like(imagen)
# #Para cargar una imagen en escala de grises:
# imagen_gray = cv2.imread("sa.png", cv2.IMREAD_GRAYSCALE)
#Para generar una imagen de grises a partir de una BGR:
# imagen_gray = cv2.imread("sa.png", cv2.IMREAD_BGR2GRAY) 

# Generar una nueva imagen
# cv2.imwrite("nueva.png", imagen_gray)

# 3. COPIAR, EXTRAER ROI Y ACCEDER A UN PIXEL  
# imagen2 = imagen.copy()
# # Si se quiere copiar solo una ROI de la imagen:
# x0 = 200
# x1 = 400
# y0 = 200
# y1 = 400
# imagen2 = imagen[x0:x1,y0:y1].copy()
# #Para leer y escribir en un pixel (x,y) se puede utilizar:
# x = 50
# y = 35
#uint8 (grises), vector[3] (RGB)
#valor_px = imagen[y,x]   # Lectura
#imagen[y,x] = valor_px # Escritura

# 4. HERRAMIENTAS DE DIBUJO

# cv2.line(imagen, start point, end point, color, thickness, line type) 
# cv2.circle(imagen, center point, radius, color, thickness, line type)
# cv2.rectangle(imagen, start point, end point, color, thickness)

    # 4.1 MOSTRAR IMÁGENES
# Para cargar una imagen en la figura 
#plt.imshow(imagen) 
# Se pueden asignar falsos colores a imágenes de grises
# plt.imshow(imagen gray, cmap= ’gray’)
#Se pueden utilizar subfiguras con plt.subplot(num filas,num columnas,actual): 
# plt.subplot(1,3,1) 
# plt.imshow(imagen) 
# plt.subplot(1,3,2) 
# plt.imshow(imagen gray, cmap = ’gray’) 
# plt.subplot(1,3,3) 
# plt.imshow(imagen2)
# plt.show()


# 5. AVANZADOS

# 5.1 Eventos del mouse

#Se define la funcion
    
def click(event, x, y, flags, param):
    #Si se presiona el click izquierdo
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        #dibuja la linea entre los puntos
        cv2.line(imagen, refPt[0], refPt[1], (0,255,0), 2)
        cv2.imshow(str_win, imagen)

str_win = "DibujaWin"    
cv2.namedWindow(str_win)
cv2.setMouseCallback(str_win,click)

while True:
    cv2.imshow(str_win, imagen)
    key = cv2.waitkey(1) & 0xFF
    if key == ord("c"):
        break

# 5.3 PASAJE DE PARÁMETROS 
    #biblioteca que permite el manejo del pasaje de parámetros al programa es argparse 
    #y se debe incluir como import argparse

# se crea el analizador de parámetros y se especifican
ap = argparse.ArgumentParser()
ap.add_argument("-ig","--imagen_gris",required=True,help="path de la imagen de grises")
ap.add_argument("-ic","--imagen_color",required=True,help="path de la imagen color")
args = vars(ap.parse_args())

# Se recuperan los parámetros en variables o directamente se usan
nombre_imagen = args["imagen_gris"]
imagen_2 = cv2.imread(args["imagen_color"])

#