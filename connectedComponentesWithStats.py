import cv2 as cv
import utils as ut
import matplotlib.pyplot as plt
celulas = cv.imread("celulas.jpg")


celulas_gris = cv.cvtColor(celulas, cv.COLOR_BGR2GRAY)
celulas_bin = cv.threshold(celulas_gris, 30, 255, cv.THRESH_BINARY)[1]


resultado = ut.encontrar_componentes_y_posiciones(celulas_bin)

print("Número de círculos:", resultado["num_componentes"])
chicos = 0 
medianos = 0
grandes = 0
for componente in resultado["componentes"]:
    print("Posición:", componente["posicion"], "Área:", componente["area"])
    if componente["area"] < 500:
        chicos += 1
    elif componente["area"] < 2000:
        medianos += 1
    else: 
        grandes += 1

print("Total chicos: ", chicos)
print("Total medianos: ", medianos)
print("Total grandes: ", grandes)

plt.imshow(celulas_bin, cmap='gray')
plt.axis('tight')
plt.show()