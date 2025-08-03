import cv2 as cv
import numpy as np
from utils import contar_hasta_cero
import matplotlib.pyplot as plt
from utils import diferencia

def bounding_box(imagen):
    indices_objeto = np.where(imagen == 255)

    # Encuentra los límites superior, inferior, izquierdo y derecho del objeto
    top = np.min(indices_objeto[0])
    bottom = np.max(indices_objeto[0])
    left = np.min(indices_objeto[1])
    right = np.max(indices_objeto[1])
    print(left,right,top,bottom)
    # esquina_superior_izquierda = [left,top]
    # esquina_inferior_derecha = [right,bottom]

    return [left,right,top,bottom]#Izquierda, derecha, arriba, abajo
ruta1 = "Imagenes_Ej/blister_completo.jpg"
ruta2 = "Imagenes_Ej/blister_incompleto.jpg"

blister_completo = cv.imread(ruta1, cv.IMREAD_GRAYSCALE)
blister_completo = blister_completo[:,:291] #por que hay un pixel de diferencia en tamaño (?)
blister_incompleto = cv.imread(ruta2, cv.IMREAD_GRAYSCALE)

cv.imshow("Completo", blister_completo)
cv.imshow("Incompleto", blister_incompleto)
cv.waitKey(0)
cv.destroyAllWindows()
#Función que reciba como parámetro la imagen del blister a analizar y devuelva un mensaje indicando si el mismo contiene
    #o no la totalidad de las pildoras. En caso de estar incompleto, indique la posición (x,y) de las pildoras faltantes

# Con un umbral de transición poner de forma tal que las pildoras queden en 255 y lo otro en 0
umbral = 100
blister_completo = np.where(blister_completo < umbral, 0, 255).astype(np.uint8)
# blister_completo = blister_completo[35:270,25:120]
blister_incompleto = np.where(blister_incompleto < umbral, 0, 255).astype(np.uint8)
# blister_incompleto = blister_incompleto[35:270,25:120] 
cv.imshow("Completo", blister_completo)
cv.imshow("Incompleto", blister_incompleto)
cv.waitKey(0)
# Se puede usar la función find countours?
diferencia = diferencia(blister_completo, blister_incompleto)
#Con la diferencia contra la imagen llena se obtienen las pildoras que faltan 

#Gráficar con plot 35-270 , 25-120
plt.figure()
plt.imshow(diferencia,cmap='gray')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True,'minor')
bordes = bounding_box(diferencia)
ancho = bordes[1] - bordes[0]
alto = bordes[3] - bordes[2] 
rectangulo = plt.Rectangle((bordes[0],bordes[2]),ancho,alto,linewidth=1,edgecolor='red',facecolor='none')
plt.gca().add_patch(rectangulo)
x = bordes[0] + ancho/2
y = bordes[2] + alto/2
posicion = [x,y]
print(f"La posición de la pildora es ({posicion})")
plt.show()
# ancho_pildoras = contar_hasta_cero()