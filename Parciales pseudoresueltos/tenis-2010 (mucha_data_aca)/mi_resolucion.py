import cv2 as cv
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

########################################################################################################
#                                         Carga de imagenes
########################################################################################################
cancha1 = cv.imread('tenis-2010 (mucha_data_aca)/tenis01.jpg')
cancha2 = cv.imread('tenis-2010 (mucha_data_aca)/tenis02.jpg')
cancha3 = cv.imread('tenis-2010 (mucha_data_aca)/tenis09.jpg')

canchas = [cancha1, cancha2, cancha3]
#############################################################################################
#                                 Pre-Procesamiento                                        
#############################################################################################
cancha = cancha3
cancha_gris = cv.cvtColor(cancha, cv.COLOR_BGR2GRAY)
print("---------------------------Pre-Procesamiento-------------------------------")
#255, 181 (obtenido por trackbars)
cancha_blurred = cv.GaussianBlur(cancha_gris, (5,5),0)
cancha_bordes = cv.Canny(cancha_blurred, 255, 181)
cancha_blurred_color = cv.GaussianBlur(cancha, (5,5),0)

cv.imshow("Imagen bordes", cancha_bordes)
cv.waitKey(0)
cv.destroyAllWindows()
#############################################################################################
#                              Segmentación de jugadores                                        
#############################################################################################
print("---------------------------Segmentación-------------------------------")
color_promedio_cancha, color_promedio_hsv = ut.color_promedio(cancha_blurred_color)
print("Color promedio BGR: ",color_promedio_cancha)
print("Color promedio HSV: ", color_promedio_hsv)
ut.trackbar_segmentacion_hsv_inverso(cancha_blurred_color)
rango_hue = [color_promedio_hsv[0]-20, color_promedio_hsv[0]+20]
rango_saturacion = [0, 255]
# Se obtienen los jugadores y otras cosas, para mejorar la segmentación hay que eliminar lo demás (erosion y dilatación?)
    #Capaz está bien porque después se pisa con lo demás y no pasa nada?
_, primera_mascara = ut.segmentacion_hsv(cancha_blurred_color,rango_hue, rango_saturacion)
primera_mascara_invertida = cv.bitwise_not(primera_mascara)
primera_segmentacion = cv.bitwise_and(cancha, cancha, mask=primera_mascara_invertida)
cv.imshow("Primera segmentación", primera_segmentacion)
cv.imshow("Primera máscara", primera_mascara_invertida)
cv.waitKey(0)
cv.destroyAllWindows()

#############################################################################################
#                                   Encontrar líneas de saque  ???                                 
#############################################################################################


ut.trackbar_umbral(cancha1_blurred_color)
ut.trackbar_sobel(cancha1_blurred_color)

#############################################################################################
#               Poner color de ancho cancha detrás de las lineas de saque                                        
#############################################################################################





#############################################################################################
#                 DE ACÁ PARA BAJO OTROS MÉTODOS (NO SIRVEN PARA ESTE CASO)                                        
#############################################################################################
# Dejar solo líneas horizontales

############################################################################################
#                          Método contorno intersecciones Hough                                        
############################################################################################
intersecciones = False
if intersecciones:
    ut.trackbar_hough_lineas(cancha_blurred_color)
    intersecciones = np.array(ut.encontrar_intersecciones(cancha_bordes, 126, 255,20))
    ut.dibujar_intersecciones(cancha1, intersecciones)
    # contornos = find_countours_intersecciones(cancha1, intersecciones)
    metodo_interseccion_aproximacion_poligono = False
    if metodo_interseccion_aproximacion_poligono:
        # Aproximar un polígono alrededor de las intersecciones
        epsilon = 10  # Parámetro de precisión para la aproximación
        approx_polygon = cv.approxPolyDP(intersecciones, epsilon, closed=True)

        # Dibujar el polígono aproximado sobre una imagen en blanco
        polygon_image = np.zeros_like(cancha1)
        cv.polylines(polygon_image, [approx_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Mostrar la imagen con el polígono aproximado
        cv.imshow('Approximated Polygon', polygon_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

#############################################################################################
#                                   Enmascarar                                        
#############################################################################################
enmascarar = False
if enmascarar:
    mascara = np.zeros_like(cancha)
    cv.rectangle(mascara, rect_top_left, rect_bottom_right, (255, 255, 255), thickness=cv.FILLED) #Rectángulo lleno
    # Si el jugador se ha detectado, restar el contorno del jugador de la máscara
    if contorno_jugador is not None:
        cv.drawContours(mascara, [contorno_jugador], -1, (0, 0, 0), thickness=cv.FILLED)
    # Aplicar la máscara a la imagen original para pintar el rectángulo sin pintar sobre el jugador
    result = cv.bitwise_and(imagen, mascara)
    # Pintar el área deseada en la imagen original usando la máscara invertida
    inverse_mask = cv.bitwise_not(mascara)
    paint_color = (0, 255, 0)  # Verde
    imagen[inverse_mask[:, :, 0] == 0] = paint_color
    # Combinar la imagen pintada con la original usando la máscara
    final_result = cv.addWeighted(result, 1, imagen, 1, 0)

    # Mostrar la imagen final
    cv.imshow('Result', final_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
#############################################################################################
#                                 Método contornos_rectángulos                                        
#############################################################################################
contornos_con_rectangulos = False
if contornos_con_rectangulos:
    contornos, _ = cv.findContours(cancha_bordes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    ut.rectangulos_trackbar(cancha1)
    # plt.imshow(cv.cvtColor(cancha1, cv.COLOR_BGR2RGB))
    # plt.show()
    area_min = 20000
    area_max = 30000
    radio = 50
    for contorno in contornos:
        x, y, w, h = cv.boundingRect(contorno)
        area = w*h
        if area_min <= area <= area_max:
            cv.rectangle(cancha1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centro_x = x + int(w/2)
            centro_y = y + int(h/2)
            #Pintar dentro del rectángulo sin pintar encima de los jugadores
            if np.linalg.norm((cancha1[centro_y,centro_x] - color_promedio_cancha),axis=-1) <= radio: 
                print("HOLA")

    cv.imshow("Cancha",cancha1)
    cv.waitKey(0)

#############################################################################################
    #                                 Otro método                                        
#############################################################################################
# contornos,_ = cv.findContours(cancha1_bordes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# imagen_con_poligonos = np.copy(cancha1)

# contornos_a_llenar = []
# for contorno in contornos:
#     #Acá hay que filtrar los contornos para solo rellenar los que queremos

# imagen_con_poligonos = llenar_poligonos(cancha1, contornos_a_llenar)
# cv.imshow("Poligonos", imagen_con_poligonos)
# cv.waitKey(0)


aproximar_poligonos = False
if aproximar_poligonos:
    if len(contornos) > 0:
        largest_contour = max(contornos, key=cv.contourArea)

        # Encontrar la línea de saque
        approx = cv.approxPolyDP(largest_contour, cv.arcLength(largest_contour, True) * 0.02, True)
        if len(approx) == 4:
            baseline = approx[0] + approx[2]  # La línea de saque es la línea horizontal que une los dos puntos inferiores

            ## Rellenar la zona de interés##
            mask = np.zeros_like(cancha1)
            cv.fillConvexPoly(mask, [baseline], (0,255,0)) #Llena un poligono convexo con un color
                                                            #el segundo parámetro es la matriz de puntos  que define el poligono, secuencia en orden
            masked_image = cv.bitwise_and(cancha1, cancha1, mask=mask)

    cv.imshow("Fin", masked_image)
