import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def mostrar_histogramas_hsv(imagen):
    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    
    # Dividir la imagen en los componentes de matiz (H), saturación (S) y valor (V)
    h, s, v = cv.split(imagen_hsv)
    
    # Calcular los histogramas de cada componente
    hist_h = cv.calcHist([h], [0], None, [180], [0, 180])
    hist_s = cv.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv.calcHist([v], [0], None, [256], [0, 256])
    
    # Mostrar los histogramas
    plt.figure(figsize=(8, 6))
    plt.subplot(311)
    plt.plot(hist_h, color='b')
    plt.title('Histograma de Matiz (H)')
    plt.xlabel('Valor de H')
    plt.ylabel('Frecuencia')
    
    plt.subplot(312)
    plt.plot(hist_s, color='g')
    plt.title('Histograma de Saturación (S)')
    plt.xlabel('Valor de S')
    plt.ylabel('Frecuencia')
    
    plt.subplot(313)
    plt.plot(hist_v, color='r')
    plt.title('Histograma de Valor (V)')
    plt.xlabel('Valor de V')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

def rgb_to_hsv(color_elegido):
    r = color_elegido[0]
    g = color_elegido[1]
    b = color_elegido[2]
    # Normalizar los valores RGB
    r /= 255.0
    g /= 255.0
    b /= 255.0
    
    # Encontrar el valor máximo y mínimo de los componentes RGB
    maxRGB = max(r, g, b)
    minRGB = min(r, g, b)
    
    # Calcular la luminosidad (V)
    V = maxRGB*100
    
    # Calcular la saturación (S)
    if maxRGB == 0:
        S = 0
    else:
        S = ((maxRGB - minRGB) / maxRGB)*100
    
    # Calcular el matiz (H)
    if maxRGB == minRGB:
        H = 0
    elif maxRGB == r:
        H = 60 * ((g - b) / (maxRGB - minRGB) % 6)
    elif maxRGB == g:
        H = 60 * ((b - r) / (maxRGB - minRGB) + 2)
    else: # maxRGB == b
        H = 60 * ((r - g) / (maxRGB - minRGB) + 4)
    
    return H, S, V

def segmentar_hsv(imagen, rango_hue, rango_saturation):
    # Paso de BGR a HSV
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    
    # obtengo H y S
    h, s, _ = cv.split(imagen_hsv)
    
    # Creo máscara
    mask = np.logical_and(np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]), 
                          np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]))
    
    # Aplicar la máscara a la imagen original (en teoría estaría en RGB)
    segmentacion = cv.bitwise_and(imagen, imagen, mask=np.uint8(mask))
    
    return segmentacion

# Segmentación: Este proceso permite separar la imagen en regiones utilizando informacion del color. 
    # En este ejercicio usted debe implementar la segmentacion de imagenes para los modelos de color RGB y HSV. 
        #En cada caso debera determinar el subespacio a segmentar para generar una mascara, que luego utilizara para extraer 
        # solo la informacion de interes de la imagen original. 
        #En cuanto a la metodologıa, le proponemos que utilice la imagen ‘futbol.jpg’ y defina una ROI representativa 
        #del color a segmentar, 
            #luego:
ruta = "Imagenes_Ej/futbol.jpg"        
imagen = cv.imread(ruta)

# • para el modelo RGB: use la informacion para calcular el centro de la esfera y su radio. Podrıa reemplazar 
    # formula de la esfera por la de una elipsoide.

# Imagen en espacio RGB
####################################################################################################################
#                                                 PRIMER EJERCICIO
####################################################################################################################
imagen_rgb = cv.cvtColor(imagen,cv.COLOR_BGR2RGB)
primer_ejercicio = False
if primer_ejercicio:
    plt.imshow(imagen_rgb)
    plt.show()

    COLOR_ELEGIDO_RGB = [40,92,201]
    # Creación de la esfera (radio y centro)
    centroide_rgb = np.array(COLOR_ELEGIDO_RGB)#COLOR ELEGIDO A OJO
    radio_rgb = 100

    # Obtenemos distancias
    distancias = np.linalg.norm(imagen_rgb - centroide_rgb, axis=-1)


    mantener_colores = False
    if mantener_colores:
        mascara_rgb = np.where(distancias <= radio_rgb, 255, 0).astype(np.uint8) #Donde las distancias sean <= radio 1 sino 0)
        imagen_rgb_segmentada = cv.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara_rgb)
    else:
        # Máscara para píxeles con distancia menor al radio
        mascara_rgb = np.where(distancias <= radio_rgb, 1, 0).astype(np.uint8)
        # Definir colores para dentro y fuera
        color_dentro_radio = (255,0,0)
        color_fuera_radio = (0,255,0)
        #Aplico máscara
        imagen_rgb_segmentada = np.where(mascara_rgb[...,None], color_dentro_radio, color_fuera_radio)

    plt.imshow(imagen_rgb_segmentada)
    plt.show()

####################################################################################################################
#                                                 SEGUNDO EJERCICIO
####################################################################################################################

# • Para el modelo HSV: Utilice las componentes H y S para determinar el subespacio rectangular a segmentar.
    # Consejo: utilizar los histogramas puede ser una buena alternativa.
segundo_ejercicio = False
if segundo_ejercicio:
    mostrar_histogramas_hsv(imagen)
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    
    COLOR_ELEGIDO_HSV = [35,25]
    print("COLOR_HSV: ", COLOR_ELEGIDO_HSV)
    ancho_hue = 10
    ancho_saturation = 50
    rango_hue = [COLOR_ELEGIDO_HSV[0] - ancho_hue, COLOR_ELEGIDO_HSV[0] + ancho_hue]
    rango_saturation = [COLOR_ELEGIDO_HSV[1] - ancho_saturation, COLOR_ELEGIDO_HSV[1] + ancho_saturation+100]
    rango_hue = [140,200]
    rango_saturation = [100,255]
    print("rango_hue: ", rango_hue)
    print("rango_saturation: ", rango_saturation)
    
    segmentacion = segmentar_hsv(imagen, rango_hue, rango_saturation)
    segmentacion_RGB = cv.cvtColor(segmentacion,cv.COLOR_BGR2RGB)
    plt.imshow(segmentacion_RGB)
    plt.show()
    # • Compare, analice y saque conclusiones sobre los resultados de ambos metodos. 
        #HSV: (Solo voy a usar información cromática, no se ve afectado por el brillo)


####################################################################################################################
#                                                 TERCER EJERCICIO
####################################################################################################################
# • Pruebe su implementacion con otras imagenes, por ejemplo segmentando solo la
# piel en las imagenes s01 i08 H CM.png, s03 i10 H DM.png, s05 i08 H LB.png, 
#s06 i13 H LV.png, s08 i06 H MA.png. Analice el desempeño de ambos metodos. 
#s06 i13 H LV.png, s08 i06 H MA.png]
tercer_ejercicio = True
if tercer_ejercicio:
    rutas = ["s01_i08_H_CM.png", "s03_i10_H_DM.png", "s05_i08_H_LB.png","s06_i13_H_LV.png", "s08_i06_H_MA.png"] 
    imagenes = []
    prefijo = "Imagenes_Ej/"
    for ruta in rutas:
        imagen = cv.imread(prefijo+ruta)
        mostrar_histogramas_hsv(imagen)
        imagenes.append(imagen)
        imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.show()

#¿Que metodo le parece mejor? ¿Es posible obtener un conjunto de valores optimo para todas las imagenes? 
#¿Es suficiente aplicar la segmentacion sin metodos de pre-procesamiento (realce o filtrado)? 
#¿Donde cree usted que estan los mayores inconvenientes? 
#¿Que condiciones observa en la escena que son homogeneas y cuales heterogeneas? 
#(distancia camara-objeto, foco, iluminacion ambiente, fondo de la escena, ubicacion y pose del sujeto,
#color de piel, vestimenta y accesorios, etc.)
# A partir de estos analisis, ¿Podrıa usted generar una lista de consideraciones utiles para generar una base de datos de imagenes?