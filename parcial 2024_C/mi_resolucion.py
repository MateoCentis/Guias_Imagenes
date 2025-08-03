import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import utils as ut
import imutils

# Para correr el código estar parado dentro de la carpeta del parcial 
    #   (por las rutas a las imagenes y las funciones que se sacan de utils)

# Centro de la imagen (se hardcodea ya que todas son del mismo tamaño)
centro_x = 200
centro_y = 200
P0 = (centro_x, centro_y)

def obtener_cuadrante(x,y):
    global centro_x, centro_y
    if y < centro_y:
        if x < centro_x:
            return 1
        else:
            return 2
    else:
        if x < centro_x:
            return 3
        else:
            return 4

def obtener_angulo_cuadrante(cuadrante_norte, cuadrante_aguja, angulo):
    if (cuadrante_norte == 1 or cuadrante_norte == 2) and (cuadrante_aguja == 4):
        return 360-angulo
    if (cuadrante_norte == 2) and cuadrante_aguja == 3:
        return 360 - angulo
    else:
        return angulo
#############################################################################################
#                                   Carga de imágenes                                        
#############################################################################################

brujula1 = cv.imread("1.png") 
brujula2 = cv.imread("2.png") 
brujula3 = cv.imread("3.png") 
brujula4 = cv.imread("4.png")

brujulas = [brujula1, brujula2, brujula3, brujula4]

#############################################################################################
#      Definición de constantes para todas las imágenes antes de recorrer las imágenes                                        
#############################################################################################
# ut.trackbar_transformacion_RGB(brujula1) # Se obtienen los parámetros para la segmentación

# Definición de kernels para dilatacíon y erosión
kernel_dilatacion = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], dtype=np.uint8)

kernel_erosion = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=np.uint8)



#############################################################################################
#                                   Solución                                        
#############################################################################################
numero_brujula = 0
# Se van a ir mostrando imágenes intermedias luego de cada operación para visualizar el proceso
for brujula in brujulas:
    
    # Se rota la imagen para que el N siempre me quede en la parte superior de la imagen
    cv.imshow("Brujula - Imagen original", brujula)

    _, mascara = ut.segmentar_RGB(brujula, 255,0,0, 146)
    cv.imshow("mascara", mascara)

    # Obtenidos el norte y la aguja se procesan para que queden solo un elemento por cada una
    mascara_dilatada = cv.dilate(mascara,kernel_dilatacion,iterations=5)
    cv.imshow("Máscara dilatada", mascara_dilatada)

    # Erosiono para solo dejarme norte y aguja
    # ut.trackbar_erosion(mascara_dilatada, kernel_erosion) # obtengo iteraciones para erosión (11)
    mascara_erosionada = cv.erode(mascara_dilatada, kernel_erosion, iterations=12)
    cv.imshow("Máscara erosionada", mascara_erosionada)

    cv.waitKey(0)
    
    # Busco posiciones de las componentes para obtener la posición del norte y la aguja 
    cant_labels, labels, stats, centroides = cv.connectedComponentsWithStats(mascara_erosionada)

    P_norte = []
    P_aguja = []
    posiciones = []
    areas = []
    for contador, (x, y, w, h, area) in enumerate(stats):
        if contador == 0:
            continue
        centro_x = x+w//2
        centro_y = y+h//2
        posicion = [centro_x, centro_y]
        areas.append(area) # Me guardo área y posición de cada componente
        posiciones.append(posicion)
    
    if areas[0] < areas[1]: #El del área menor será el del norte
        P_norte = posiciones[0]
        P_aguja = posiciones[1]
    else:
        P_aguja = posiciones[1]
        P_norte = posiciones[0]

    # Con las posiciones armo vectores con el punto central y se miden los ángulos
    print("Norte: ", P_norte)
    print("Aguja: ", P_aguja)
    cuadrante_norte = obtener_cuadrante(P_norte[0], P_norte[1])
    cuadrante_aguja = obtener_cuadrante(P_aguja[0], P_aguja[1])

    vector_norte = [P_norte[1] - P0[1], P_norte[0] - P0[0]] # N - P0
    vector_aguja = [P_aguja[1] - P0[1], P_aguja[0] - P0[0]] # Aguja - P0

    producto_punto_unitario = np.dot(vector_aguja,vector_norte)/(np.linalg.norm(vector_aguja)*np.linalg.norm(vector_norte))
    angulo = math.acos(producto_punto_unitario)*(180/math.pi) 

    # Segun el cuadrante obtener el ángulo, midiendo siempre en contra reloj
    angulo = obtener_angulo_cuadrante(cuadrante_norte, cuadrante_aguja, angulo)
    # Otras pruebas
    # angulo = np.arctan2(vector_norte[1]-P0[1], vector_aguja[1]-P0[1]) - np.arctan2(vector_norte[0]-P0[0], vector_norte[0]-P0[0])*(180/math.pi) 

    numero_brujula += 1
    print("El ángulo de la Brújula ", str(numero_brujula)," es: ", angulo)




#############################################################################################
#                                   Otra solución explicada en el informe
#############################################################################################
otra_solucion = False
if otra_solucion:
    for brujula in brujulas:

        plt.figure(),plt.imshow(cv.cvtColor(brujula,cv.COLOR_BGR2RGB)), plt.show()
        brujula_segmentada, mascara = ut.segmentar_RGB(brujula, 255,0,0, 146)
        cv.imshow("mascara", mascara)

        # Obtenidos el norte y la aguja se procesan para que queden solo un elemento por cada una
        mascara_dilatada = cv.dilate(mascara,kernel_dilatacion,iterations=5)
        cv.imshow("Dilatada", mascara_dilatada)

        # Erosiono para solo dejarme norte y aguja
        # ut.trackbar_erosion(mascara_dilatada, kernel_erosion) # obtengo iteraciones para erosión (11)
        mascara_erosionada = cv.erode(mascara_dilatada, kernel_erosion, iterations=12)
        cv.imshow("Erosionada", mascara_erosionada)
        cv.imshow("Brujula", brujula)
        cv.waitKey(0)
        
        # Busco posición de norte y aguja 
        cant_labels, labels, stats, centroides = cv.connectedComponentsWithStats(mascara_dilatada)

        #############################################################################################
        #                         Aquí la diferencia con el otro método                                        
        #############################################################################################
        P_norte = []
        P_aguja = []
        for contador, (x, y, w, h, area) in enumerate(stats):
            if contador == 0:
                continue
            elif contador == 1: #norte 
                centro_norte_x = x+w//2
                centro_norte_y = y+h//2
                P_norte = [centro_norte_x, centro_norte_y]
            else: #aguja
                centro_aguja_x = x+w//2
                centro_aguja_y = y+h//2
                P_aguja = [centro_aguja_x, centro_aguja_y]
        #############################################################################################
        #                                                                           
        #############################################################################################
        # Con las posiciones armo vectores con el punto central y se miden los ángulos
        print("Norte: ", P_norte)
        print("Aguja: ", P_aguja)
        vector_norte = [P_norte[1] - P0[1], P_norte[0] - P0[0]] # N - P0
        vector_aguja = [P_aguja[1] - P0[1], P_aguja[0] - P0[0]] # aguja - P0

        producto_punto_unitario = np.dot(vector_aguja,vector_norte)/(np.linalg.norm(vector_aguja)*np.linalg.norm(vector_norte))
        # Según el cuadrante cambio el ángulo
        angulo = math.acos(producto_punto_unitario)*(180/math.pi)
        
        print("Ángulo Brújula ", str(contador+1),": ", angulo)

