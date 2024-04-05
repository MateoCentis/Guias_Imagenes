import cv2 as cv
import numpy as np
from utils import ventana_trackbars
# Kernel de convolucion h(s, t): matriz (filtro) que es utilizado para realizar la con-
    # volucion. (tamanos habituales: 3 × 3 y 5 × 5)
# Mascara de filtrado w(s, t): matriz que se utiliza para realizar la correlacion. 
    # Corresponde a un kernel de convolucion rotado 180◦

#FILTROS PASA BAJOS

# 1. Genere diferentes mascaras de promediado, utilizando ﬁltro de promediado o
    # caja (box ﬁlter) y el formato cruz.
# Aplique los ﬁltros sobre una imagen y veriﬁque los efectos de aumentar el
    # tamano de la mascara en la imagen resultante.
# Ayuda: mask = np.ones((3,3),np.float32)/9
mascara = np.ones((3,3),np.float32)/9
mascara_cruz = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)


# Cargar la imagen
ruta = "Imagenes_Ej/imagenA.tif"
imagen = cv.imread(ruta, cv.IMREAD_GRAYSCALE)

# Crear una máscara de promediado (filtro de la media) 3x3
sizeMascara = 3
mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)

# Aplicar la convolución con la máscara de promediado
imagen_mascara_promedio = cv.filter2D(imagen, -1, mascara_promedio)

# Crear una máscara en formato cruz 3x3
mascara_cruz = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

# Aplicar la máscara en formato cruz
imagen_mascara_cruz = cv.filter2D(imagen, -1, mascara_cruz)

# Mostrar las imágenes resultantes
cv.imshow("Original Image", imagen)
cv.imshow("Imagen mascara promedio", imagen_mascara_promedio)
cv.imshow("Imagen mascara cruz", imagen_mascara_cruz)
cv.waitKey(0)
cv.destroyWindow("Imagen mascara promedio")
cv.destroyWindow("Imagen mascara cruz")


# 2. Genere mascaras de ﬁltrado gaussianas con diferente σ y diferente tamaño.
# Visualice y aplique las mascaras sobre una imagen. Compare los resultados
# con los de un ﬁltro de promediado del mismo tamaño.

# Crear máscaras de suavizado gaussiano con diferentes σ
sigma_values = [0.5,1,2]
gaussian_filtered_images = []

for sigma in sigma_values:
    gaussian_filtered = cv.GaussianBlur(imagen,(sizeMascara, sizeMascara),sigma)
    gaussian_filtered_images.append(gaussian_filtered)

# Mostrar las imágenes resultantes
for i, sigma in enumerate(sigma_values):
    cv.imshow(f"Gaussian ( sigma = {sigma} )", gaussian_filtered_images[i])
cv.waitKey(0)
cv.destroyWindow("Gaussian ( sigma = 0.5 )")
cv.destroyWindow("Gaussian ( sigma = 1 )")
cv.destroyWindow("Gaussian ( sigma = 2 )")



# 3. Utilice el ﬁltro de mediana sobre una imagen con diferentes tamaños de ven-
# tana. Compare los resultados con los ﬁltros anteriores para un mismo tamaño.
filtrado_mediana = cv.medianBlur(imagen, sizeMascara)

# Mostrar la imagen resultante
cv.imshow(f"Median Filter (Window Size {sizeMascara})", filtrado_mediana)

cv.waitKey(0)
cv.destroyAllWindows()


# 4. Los ﬁltros pasa-bajos pueden utilizarse para localizar objetos grandes en una
# escena. Aplique este concepto a la imagen ’hubble.tif’ y obtenga una ima-
# gen de grises cuyos objetos correspondan solamente a los de mayor tama˜no
# de la original

ruta = "Imagenes_Ej/hubble.tif"
hubble = cv.imread(ruta,cv.IMREAD_GRAYSCALE)

variables_trackbar = ["sizeVentana","umbral"]
parametros_trackbar = [[100,1500],[1,255]]
# Se debe desenfocar y luego umbralizar 
def transformacion(imagen, valores_trackbar):
    #dividir por 100 los valores_trackbar
    sizeMascara = int(valores_trackbar[0]/100)
    umbral = valores_trackbar[1]
    if sizeMascara < 1:
        sizeMascara = 1
    if umbral < 1:
        umbral = 1
    print("Umbral: ", umbral)
    print("SizeMascara: ", sizeMascara)
    mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
    imagen_transformada = cv.filter2D(imagen, -1, mascara_promedio)
    #Umbralizar 
    _,imagen_salida = cv.threshold(imagen_transformada,umbral, 255, cv.THRESH_BINARY)
    return imagen_salida

ventana_trackbars(hubble, variables_trackbar, parametros_trackbar, transformacion)
