import cv2
import numpy as np

# Leer las imágenes
img_a7v600_x = cv2.imread('a7v600-x.gif', cv2.IMREAD_GRAYSCALE)
img_a7v600_se = cv2.imread('a7v600-SE.gif', cv2.IMREAD_GRAYSCALE)
img_a7v600_x_noisy = cv2.imread('a7v600-x(RImpulsivo).gif', cv2.IMREAD_GRAYSCALE)
img_a7v600_se_noisy = cv2.imread('a7v600-SE(RImpulsivo).gif', cv2.IMREAD_GRAYSCALE)

# Preprocesamiento de imágenes
def preprocess_image(img):
    # Aplicar cualquier preprocesamiento necesario (eliminación de ruido, realce de características, etc.)
    # Aquí se puede agregar filtrado de ruido, realce de bordes, etc.
    # Por ejemplo, aplicar un filtro de mediana para eliminar ruido impulsivo:
    img_filtered = cv2.medianBlur(img, 5)
    return img_filtered

# Extracción de características
def extract_features(img):
    # Aquí se pueden calcular características basadas en el histograma de intensidad, por ejemplo
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    return hist.flatten()

# Calcular características para cada imagen
hist_a7v600_x = extract_features(preprocess_image(img_a7v600_x))
hist_a7v600_se = extract_features(preprocess_image(img_a7v600_se))
hist_a7v600_x_noisy = extract_features(preprocess_image(img_a7v600_x_noisy))
hist_a7v600_se_noisy = extract_features(preprocess_image(img_a7v600_se_noisy))

# Comparar características de las imágenes con ruido y sin ruido
# Por ejemplo, se puede calcular la similitud de los histogramas usando la distancia chi-cuadrado
similarity_noisy_x = cv2.compareHist(hist_a7v600_x, hist_a7v600_x_noisy, cv2.HISTCMP_CHISQR)
similarity_noisy_se = cv2.compareHist(hist_a7v600_se, hist_a7v600_se_noisy, cv2.HISTCMP_CHISQR)

# Determinar la clase de placa madre en función de la similitud
if similarity_noisy_x < similarity_noisy_se:
    print("La imagen corresponde a la placa madre A7V600-X.")
else:
    print("La imagen corresponde a la placa madre A7V600-SE.")
