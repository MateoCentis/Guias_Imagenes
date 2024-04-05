import cv2 as cv
import numpy as np

def nothing(x):
    pass
def evitar_desborde(imagen):

    minimo = min(imagen)
    maximo = max(imagen)

    if minimo < 0:
        imagen = imagen + 255
        imagen = imagen / 2
    if maximo > 255:
        imagen = (imagen - minimo)*(255/(maximo-minimo))
    
    return imagen

########################################################################################################
# 1. Implemente una función que realice las siguientes operaciones
    # sobre dos imagenes que sean pasadas como parametros:
########################################################################################################

ruta1 = "imagenes_varias/pikachu.jpg"
ruta2 = "imagenes_varias/sentarse.jpg"
imagen1 = cv.imread(ruta1,cv.IMREAD_GRAYSCALE)
imagen2 = cv.imread(ruta2,cv.IMREAD_GRAYSCALE)


#a) Suma. Normalice el resultado por el numero de imagenes. 
def suma(imagen1,imagen2):
    alpha = 0.5
    return (1-alpha)*imagen1 + alpha*imagen2

#Para hacer suma pesada usar blending
#cv.blendLinear()
#b) Diferencia. Aplique las dos funciones de reescalado usadas tıpicamente 
    #para evitar el desborde de rango (sumar 255 y dividir por 2, o restar el mınimo y escalar a 255).
def diferencia(imagen1,imagen2):
    diferencia_imagenes = imagen1 - imagen2
    return evitar_desborde(diferencia_imagenes)
#c) Multiplicación. En esta operacion la segunda imagen debera ser una mascara binaria, 
    #muy utilizada para la extraccion de la region de interes (ROI) de una imagen.

def multiplicacion(imagen1,imagen2):#imagen2: debe ser una mascara binaria
    return imagen1 * imagen2

# Prueba de operaciones..

########################################################################################################
# 2. A partir de un video (pedestrians.mp4) de una camara de seguridad, debe
    #obtener solamente el fondo de la imagen. Incorpore un elemento TrackBar que le permita ir eligiendo 
    #el numero de frames a promediar para observar los resultados instantaneamente.
########################################################################################################

def calcular_frame_promedio(frames,numero_imagenes):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_suma = np.zeros((height, width, 3), dtype=np.float32)
    cont = 0 
    for frame in frames[:numero_imagenes]:
        # if cont == numero_imagenes:
        #     break 
        frame_suma += frame
        cont += 1
    frame_promedio = (frame_suma / cont).astype(np.uint8) #numero_imagenes == cont?
    return frame_promedio

NOMBRE_VENTANA = "Fotografia resultado"

cv.namedWindow(NOMBRE_VENTANA)
cap = cv.VideoCapture("Imagenes_Ej/pedestrians.mp4")

max_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
# print(frames[0].shape) #576,768,3
numero_imagenes = 1

cv.createTrackbar("NumImgs", NOMBRE_VENTANA, 1, max_frames, nothing)

imagen_resultado = np.zeros((width,height,3),dtype=np.float32)

while True:
    # Obtener cantidad de trackbar
    numero_imagenes = (cv.getTrackbarPos("NumImgs", NOMBRE_VENTANA))
    
    # Obtener imagen del video y mostrar
    imagen_resultado = calcular_frame_promedio(frames,numero_imagenes)

    cv.imshow(NOMBRE_VENTANA, imagen_resultado)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

