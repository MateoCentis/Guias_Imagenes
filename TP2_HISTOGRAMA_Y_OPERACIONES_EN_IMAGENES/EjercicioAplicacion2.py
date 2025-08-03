import numpy as np
import cv2 as cv

################################################################################################
#FUNCIONES
################################################################################################
def calcular_MSE(img1, img2):
    error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    error /= float(img1.shape[0] * img1.shape[1])
    return error
def evitar_desborde(imagen):

    minimo = np.min(imagen)
    maximo = np.max(imagen)

    if minimo < 0:
        imagen = imagen + 255
        imagen = imagen / 2
    if maximo > 255:
        imagen = (imagen - minimo)*(255/(maximo-minimo))
    
    return imagen
def diferencia(imagen1,imagen2):
    diferencia_imagenes = imagen1 - imagen2
    return evitar_desborde(diferencia_imagenes)
def obtener_promedio_video(cap):
    ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_suma = np.zeros((alto,ancho,3), dtype=np.float32)
    cont = 0
    while True:
        ret, frame = chipSE_ruido.read()

        if not ret:
            break
        frame_suma += frame
        cont += 1

    frame_promedioSE_ruido = frame_suma/cont
    return frame_promedioSE_ruido

def mostrar_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow(f'Video {cap}', frame)
    cv.waitKey(0)


ruta1 = "Imagenes_Ej/a7v600-SE.gif"
ruta2 = "Imagenes_Ej/a7v600-X.gif"

ruta1_ruido = "Imagenes_Ej/a7v600-SE(RImpulsivo).gif"
ruta2_ruido = "Imagenes_Ej/a7v600-X(RImpulsivo).gif"

################################################################################################
#VISUALIZACIÓN DE LOS DATOS
################################################################################################
chipSE = cv.VideoCapture(ruta1)
chipX = cv.VideoCapture(ruta2)

chipSE_ruido = cv.VideoCapture(ruta1_ruido)

chipX_ruido = cv.VideoCapture(ruta2_ruido)

_,frameSE = chipSE.read()
_,frameX = chipX.read()
frameSE = cv.cvtColor(frameSE, cv.COLOR_BGR2GRAY)
frameX = cv.cvtColor(frameX, cv.COLOR_BGR2GRAY)

_,frameSE_ruido = chipSE_ruido.read()
_,frameX_ruido = chipX_ruido.read()

frameSE_ruido = cv.cvtColor(frameSE_ruido, cv.COLOR_BGR2GRAY)
frameX_ruido = cv.cvtColor(frameX_ruido, cv.COLOR_BGR2GRAY)

cv.imshow("FRAME SE", frameSE)
cv.imshow("Frame x", frameX)

cv.waitKey(0)

################################################################################################
#OBTENCIÓN DE LAS MÁSCARAS 
################################################################################################

diferencia_imagenes_sin_ruido = diferencia(frameSE,frameX)

#Se podría aplicar esta diferencia como una máscara para obtener de forma separada la parte de los chips que quiero
cv.imshow("Diferencia", diferencia_imagenes_sin_ruido)

diferencia_imagenes_con_ruido = diferencia(frameSE_ruido,frameX_ruido)
cv.imshow("Diferencia con ruido", diferencia_imagenes_con_ruido)

cv.waitKey(0)

cv.destroyAllWindows()

################################################################################################
# POSTPROCESAR Y APLICAR LAS MÁSCARAS
################################################################################################

#Máscara con 1 donde hay diferencia y 0 donde no la hay

mascara_sin_ruido = np.where(diferencia_imagenes_sin_ruido > 20, 0,diferencia_imagenes_sin_ruido).astype(np.uint8)
mascara_sin_ruido = np.where(diferencia_imagenes_sin_ruido != 0, 255, 0).astype(np.uint8) 

mascara_con_ruido = np.where(diferencia_imagenes_con_ruido > 20, 0,diferencia_imagenes_con_ruido).astype(np.uint8)
mascara_con_ruido = np.where(diferencia_imagenes_con_ruido != 0, 255, 0).astype(np.uint8) 

cv.imshow("Mascara con ruido", mascara_con_ruido)
cv.imshow("Mascara sin ruido", mascara_sin_ruido)

cv.waitKey(0)

cv.destroyAllWindows()

chipSE_resultante_sin_ruido = cv.bitwise_and(frameSE, frameSE, mask=mascara_sin_ruido)
chipX_resultante_sin_ruido = cv.bitwise_and(frameX, frameX, mask=mascara_sin_ruido)

cv.imshow("CHIP SE resultante",chipSE_resultante_sin_ruido)
cv.imshow("CHIP X resultante",chipX_resultante_sin_ruido)

cv.waitKey(0)

chipSE_resultante_con_ruido = cv.bitwise_and(frameSE_ruido,frameSE_ruido,mask=mascara_con_ruido)
chipX_resultante_con_ruido = cv.bitwise_and(frameX_ruido,frameX_ruido,mask=mascara_con_ruido)

# ojo el ruido
cv.imshow("CHIP SE resultante con ruido",chipSE_resultante_con_ruido)

cv.imshow("CHIP X resultante con ruido",chipX_resultante_con_ruido)

cv.waitKey(0)

################################################################################################
# DETERMINAR A CUAL CORRESPONDE
###############################################################################################
# Cómo lo hago? Si uso el error cuadrático comparando la máscara con cada imagen estaría bien?