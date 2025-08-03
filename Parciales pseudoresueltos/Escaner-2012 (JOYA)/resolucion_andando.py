import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_skew_angle_hough(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de desenfoque
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detectar bordes usando Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detectar líneas usando la transformada de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Calcular el ángulo promedio de las líneas detectadas
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        angles.append(angle)
    
    # Devolver el ángulo promedio
    return np.mean(angles)

# Función para rotar la imagen
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Cargar la imagen
image_path = 'Escaner-2012 (JOYA)/escaneo5.png'
image = cv2.imread(image_path)

# Obtener el ángulo de inclinación
angle = get_skew_angle_hough(image)
print(f"Ángulo de inclinación detectado: {angle} grados")

# Rotar la imagen para corregir la inclinación
corrected_image = rotate_image(image, angle)


# Mostrar la imagen original y la imagen corregida
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Imagen Corregida')
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.show()

