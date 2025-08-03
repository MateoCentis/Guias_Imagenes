import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import utils as ut


plato1 = cv.imread("Moscas-2011 (JOYA)/Platos0.jpg")
plato2 = cv.imread("Moscas-2011 (JOYA)/Platos01.jpg")
plato3 = cv.imread("Moscas-2011 (JOYA)/Platos02.jpg")
plato4 = cv.imread("Moscas-2011 (JOYA)/Platos03.jpg")
plato5 = cv.imread("Moscas-2011 (JOYA)/Platos04.jpg")

platos = [plato1, plato2, plato3, plato4, plato5]


radio_plato = 430
radio_sopa = 315

# ut.trackbar_diferentes_umbrales(plato1)
plt.imshow(plato1)
plt.show()
def es_sopa_zapallo(plato):
    rango_hue = [9,13]
    rango_saturation = [154,255]
    _, mascara_zapallo = ut.segmentacion_hsv(plato,rango_hue, rango_saturation)
    area_zapallo = np.sum(mascara_zapallo == 255)

    rango_hue_casa = [13,38]
    rango_saturation_casa = [51,214]
    _, mascara_de_la_casa = ut.segmentacion_hsv(plato, rango_hue_casa, rango_saturation_casa)
    area_casa = np.sum(mascara_de_la_casa == 255)

    if area_zapallo > area_casa:
        return True
    else:
        return False

def mostrar_tipo_sopa(tipo):
    if tipo == True:
        print("Sopa de zapallo")
    else:
        print("Sopa de la casa")

kernel_dilate = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]], dtype=np.uint8)

kernel_erode = np.array([
                        [0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=np.uint8)

fila_central = plato1.shape[0]//2
for plato in platos:
    print("------------------------------------------------------------------------")
    #############################################################################################
    #              # 1. Que tipo de sopa tiene el plato: zapallo o de la casa.                                  
    #############################################################################################
    print("1. ¿Que tipo de sopa tiene el plato: zapallo o de la casa?-------------")
    tipo_sopa = es_sopa_zapallo(plato)
    mostrar_tipo_sopa(tipo_sopa)
    #############################################################################################
    #                             2. Cuantas hay en la escena                                              
    #############################################################################################
    umbral = 54
    plato_gris = cv.cvtColor(plato, cv.COLOR_BGR2GRAY)
    moscas = ut.umbralizado_inv(plato_gris, [umbral])
    # Uno formas y luego erosiono
    moscas_dilatada = cv.dilate(moscas,kernel_dilate, iterations=13)
    moscas_erosionadas = cv.erode(moscas_dilatada,kernel_erode, iterations=13)
    cantidad, labels = cv.connectedComponents(moscas_erosionadas)
    print("2. ¿Cuántas moscas hay en la escena?-------------------")
    print(cantidad-1)
    #############################################################################################
    #                     3 y 4. Cuantas hay en el plato y cuántas en la sopa.                                        
    #############################################################################################
    # Encontrar los círculos de la imagen
    print("3. Cuántas hay en el plato------------------------")
    # Encontrar el radio de cada plato y su centro
        #tiro un rayo en la mitad de la imagen y miro
    plato_bin = cv.threshold(plato_gris, 194,255, cv.THRESH_BINARY)[1]
    perfil_horizontal = plato_bin[fila_central,:]
    for i in range(len(perfil_horizontal)):
        if perfil_horizontal[i] != 0:#llegue al borde del plato (ese i es el que necesito)
            centro_sopa = [fila_central, i+radio_plato]
            break
    
    # circulos = cv.HoughCircles(plato_bin, cv.HOUGH_GRADIENT, 7, 21, None, 25, 92, 275, 455)

    # circulos = np.round(circulos[0, :]).astype("int")
    # centro_plato = circulos[0,0:2]
    # radio_plato = circulos[0,2]
    # centro_sopa = circulos[1,0:2]
    # radio_sopa = circulos[1,2]
    # Obtener posiciones de las moscas
    posiciones_moscas = []

    # Recorrer labels encontrados (sin contar el 0)
    for label in range(1, cantidad):
        # Máscara
        mask = np.uint8(labels == label)
        # Momentos de la máscara
        moments = cv.moments(mask)
        # Porquería de momentos
        if moments["m00"] != 0:
            # Centroide de la mosca (posicion)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            
            posiciones_moscas.append((cX, cY))
        
    # np.argwhere(etiquetas == label).mean()
    # Recorro cada posición de la mosca y me fijo por el radio del círculo
    contador_sopa = 0
    contador_plato = 0
    contador_afuera = 0
    for posicion in posiciones_moscas:
        #Distancia al centro
        distancia = np.sqrt( (posicion[0]-centro_sopa[0])**2 + (posicion[1]-centro_sopa[1])**2 )
        if distancia < radio_sopa:
            contador_sopa += 1
        elif distancia > radio_sopa and distancia < radio_plato:
            contador_plato += 1
        else:
            contador_afuera += 1
    
    print("Total sopa: ", contador_sopa)
    print("Total plato: ", contador_plato)
    print("Total afuera: ", contador_afuera)
    #############################################################################################
    #            5. Finalmente, el sistema informar´a que un plato est´a bien servido si:
                    # Contiene sopa de zapallo y una cantidad m´axima de 3 moscas.
                    # Contiene sopa de la casa y una cantidad m´axima de 4 moscas.                                        
    #############################################################################################   
    if tipo_sopa and contador_sopa < 4: #zapallo
        print("Plato bien servido")
    elif not(tipo_sopa) and contador_sopa < 5:
        print("Plato bien servido")
    else:
        print("Plato mal servido")

    cv.imshow("Plato",plato)
    cv.waitKey(0)
