import cv2 as cv
import numpy as np

################################################################################################################################
# APARTADO DE FUNCIONES
################################################################################################################################
def logaritmica(r):
    global c 
    return c*np.uint8(np.log1p(imagen) * 255 / np.log1p(255))

def potencia(r, gamma):
    global c
    return c*np.power(r, gamma)

def nothing(x):
    pass

def multiplicacion(imagen1,imagen2):#imagen2: debe ser una mascara binaria
    return imagen1 * imagen2

def suma(imagen1,imagen2):
    alpha = 0.5
    return (1-alpha)*imagen1 + alpha*imagen2

# ENUNCIADO:
    #Utilizando las tecnicas aprendidas, descubra que objetos no estan perceptibles en la imagen earth.bmp 
        #y realce la imagen de forma que los objetos se vuelvan visibles con buen contraste sin realizar modificaciones
        # sustanciales en el resto de la imagen

    # Realzar contranste en la parte que es todo negro
        #Para esto hay que hacer una máscara binaria para todo lo negro negro y aplicarle la corrección solo a esa parte
            #Primero obtenemos con máscara binaria la imagen con solo lo negro
            #despues se lo sumamos a la imagen? (veremos)

################################################################################################################################
# RESOLUCIÓN
################################################################################################################################
ruta = "Imagenes_Ej/earth.bmp"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

# -Imagen sin cambios
cv.imshow("Imagen", imagen)
cv.waitKey(0)

# Operaciones a la mascara
mascara = np.zeros_like(imagen)

mascara = np.where(imagen == 0, 255, 0).astype(np.uint8) #igualar a 1 todos los (i,j) de mascara donde para ese (i,j) imagen sea igual a 0

cv.imshow("Mascara",mascara)
#aplicarle efecto de contraste o brillo a la máscara para que se vea lo oculto
c = 1
mascara_imagen_corregida = logaritmica(mascara) + 100
# print(np.sum(mascara_imagen_corregida))
cv.imshow("Máscara corregida", mascara_imagen_corregida)
cv.waitKey(0)

ancho = imagen.shape[0]
alto = imagen.shape[1]
imagen_final = np.zeros_like(imagen)
for i in range(ancho):
    for j in range(alto):
        if imagen[i,j] < 10:
            imagen_final[i,j] = mascara_imagen_corregida[i,j]
        else:
            imagen_final[i,j] = imagen[i,j]
# imagen_final = np.where(mascara_imagen_corregida > 0, imagen, mascara_imagen_corregida)
cv.imshow("Imagen final",imagen_final)
cv.waitKey(0)

# NOMBRE_VENTANA = "Logaritmica y potenscia"
# cv.namedWindow(NOMBRE_VENTANA)

# cv.createTrackbar("gamma", NOMBRE_VENTANA, 10, 100, nothing)
# cv.createTrackbar("c",NOMBRE_VENTANA,0,255,nothing)
# while True:
#     gamma = cv.getTrackbarPos("gamma", NOMBRE_VENTANA) / 20
#     c = cv.getTrackbarPos("c",NOMBRE_VENTANA)
#     # Aplicar la transformación logarítmica y de potencia
#     imagen_salida_logaritmica = np.clip(logaritmica(imagen), 0, 255).astype(np.uint8)
#     imagen_salida_potencia = np.clip(potencia(imagen, gamma), 0, 255).astype(np.uint8)

#     # Concatenar imágenes horizontalmente
#     imagen_resultado = np.hstack((imagen_salida_logaritmica, imagen_salida_potencia))

#     cv.imshow(NOMBRE_VENTANA, imagen_resultado)

#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv.destroyAllWindows()

