import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def contar_hasta_cero(arreglo):
    elementos_hasta_cero = 0

    for elemento in arreglo:
        if elemento != 0:
            elementos_hasta_cero += 1
        else:
            break

    return elementos_hasta_cero
def contar_valores_seguidos(arreglo, valor):
    cantidad_valores_seguidos = 0
    valores_actuales = 0

    for elemento in arreglo:
        if elemento == valor:
            valores_actuales += 1
            cantidad_valores_seguidos = max(cantidad_valores_seguidos, valores_actuales)
        else:
            valores_actuales = 0

    return cantidad_valores_seguidos
def contar_ceros_y_no_ceros_seguidos(arreglo):
    cantidad_ceros_seguidos = 0
    cantidad_no_ceros_seguidos = 0
    ceros_actuales = 0
    no_ceros_actuales = 0

    for elemento in arreglo:
        if elemento == 0:
            ceros_actuales += 1
            cantidad_no_ceros_seguidos = max(cantidad_no_ceros_seguidos, no_ceros_actuales)
            no_ceros_actuales = 0
        else:
            no_ceros_actuales += 1
            cantidad_ceros_seguidos = max(cantidad_ceros_seguidos, ceros_actuales)
            ceros_actuales = 0

    cantidad_ceros_seguidos = max(cantidad_ceros_seguidos, ceros_actuales)
    cantidad_no_ceros_seguidos = max(cantidad_no_ceros_seguidos, no_ceros_actuales)

    return cantidad_ceros_seguidos, cantidad_no_ceros_seguidos

ruta = "Imagenes_Ej/botellas.tif"
imagen = cv.imread(ruta,cv.IMREAD_GRAYSCALE)
print("Dimensiones: ", imagen.shape) #(196,285,3?)

# Obtención de información de la imagen a través de gráficas para determinar procedimiento
filaN = 190
columnaM = 150

plt.plot(imagen[filaN,:])
plt.grid(True)
plt.title(f"Perfil de intensidad fila {filaN}")
plt.xlabel('Posición a lo largo del segmento')
plt.ylabel('Intensidad')

plt.figure()
plt.plot(imagen[:,columnaM])
plt.grid(True)
plt.title(f"Perfil de intensidad columna {columnaM}")
plt.xlabel('')

# Encontrar ancho de botella y ancho espaciado
espacio_entre_botellas, ancho_botella = contar_ceros_y_no_ceros_seguidos(imagen[filaN,40:150])
print(f"Espacio entre botellas: {espacio_entre_botellas}")
print(f"ancho botella: {ancho_botella}")
#Recorro horizontalmente las botellas, cuando encuentro algo != 0 estoy en una botella, ahí miro verticalmente desde el centro de ella
porcentajes_llenado = []
posiciones_botellas_esquina_inferior = []
cantidad_llenada = 0
espacio_sin_llenar = 0
columna = 0
while columna < imagen.shape[1]: #recorro todas las columnas
    print(f"COLUMNA: {columna}")
    #en la fila 190 (una de las más abajo, se toma como piso, después podría probar con la última) [ya que si está mas vacía que eso chau]
    intensidad = imagen[190,columna]
    if intensidad > 1: #estoy en una botella
    #Pueden pasar tres casos:
        
        #1. El gráfico arranca con una botella
        if columna - ancho_botella + 10 < 0: 
            botella_restante = contar_hasta_cero(imagen[190,:])
            if botella_restante > ancho_botella/2: #si queda más de media botella (moverlo un poco más adelante a ojo)
                _, cantidad_llenada = contar_ceros_y_no_ceros_seguidos(imagen[:,columna+int(ancho_botella/4)])
                espacio_sin_llenar = contar_valores_seguidos(imagen[:,columna+int(ancho_botella/4)],255)
            else: #Mido ahí nomás porque es lo que hay
                _, cantidad_llenada = contar_ceros_y_no_ceros_seguidos(imagen[:,columna])
                espacio_sin_llenar = contar_valores_seguidos(imagen[:,columna],255)
            porcentajes_llenado.append(cantidad_llenada/(cantidad_llenada+espacio_sin_llenar))
            espaciado = botella_restante
            print("ARRANCA")
        
        #2. El gráfico termina con una botella
        elif columna + ancho_botella > imagen.shape[1]:
            botella_restante = contar_hasta_cero(imagen[190,::-1]) #Lo invierto al arreglo
            if columna + ancho_botella/2 < botella_restante: #tengo margen para medir la mitad de la botella
                _, cantidad_llenada = contar_ceros_y_no_ceros_seguidos(imagen[:,columna+int(ancho_botella/2)])
                espacio_sin_llenar = contar_valores_seguidos(imagen[:,int(columna+ancho_botella/2)],255)
            else: #mido ahí nomás 
                _, cantidad_llenada = contar_ceros_y_no_ceros_seguidos(imagen[:,columna])
                espacio_sin_llenar = contar_valores_seguidos(imagen[:,columna],255)
            porcentajes_llenado.append(cantidad_llenada/(cantidad_llenada+espacio_sin_llenar))
            print("TERMINA")
            posiciones_botellas_esquina_inferior.append(columna)
            # Salgo del for (no hay espaciado)
            break
        #3. La botella está en el medio del gráfico(medimos altura para la columna del medio de la botella)
        else:
            mitad_botella = columna + int(ancho_botella/2)
            _, cantidad_llenada = contar_ceros_y_no_ceros_seguidos(imagen[:,mitad_botella])
            espacio_sin_llenar = contar_valores_seguidos(imagen[:,mitad_botella],255)
            porcentajes_llenado.append(cantidad_llenada/(cantidad_llenada+espacio_sin_llenar))
            print("MEDIO")
            espaciado = ancho_botella 
        print(f"ESPACIADO: {espaciado}")
        posiciones_botellas_esquina_inferior.append(columna)
        columna += espaciado
        continue
    columna += 1


print(f"Porcentajes de llenado: {porcentajes_llenado}")
print(f"Posiciones: {posiciones_botellas_esquina_inferior}")

plt.figure()
plt.imshow(imagen)
for posicion in posiciones_botellas_esquina_inferior:
    plt.plot([posicion,posicion], [0, imagen.shape[0] - 1])


plt.show()


#Con las posiciones de cada botella y su porcentaje de llenado recuadro las que no están llenas
lleno = max(porcentajes_llenado)
print(f"LLeno: {lleno}")
# Se dibuja un recuadro
for i in range(len(porcentajes_llenado)):
    porcentaje = porcentajes_llenado[i]
    if porcentaje + 0.05 < lleno: #No llena

        x0 = posiciones_botellas_esquina_inferior[i]
        x1 = x0 + ancho_botella
        y0 = 0 + 2
        y1 = imagen.shape[0]  - 2

        print(f"[x0,y0]: {x0,y0}") # (126,0)
        print(f"[x1,y1]: {x1,y1}") # (180,196)
        #dibujarle un rectángulo
        cv.rectangle(imagen, (x0,y0), (x1,y1), (255,0,0), 2)
        # Guardar la imagen modificada
        cv.imwrite('TP1/imagen_con_recuadros.tif', imagen)
        print(f"PORCENTAJE DE LLENADO BOTELLA VACÍA: {porcentaje}")


imagen_con_recuadros = cv.imread('TP1/imagen_con_recuadros.tif')
cv.imshow("Con recuadros", imagen_con_recuadros)
cv.waitKey(0)
cv.destroyAllWindows()