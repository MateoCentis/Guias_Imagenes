import cv2 as cv
import numpy as np

# Función para detectar líneas utilizando la transformada de Hough
def detect_lines(image, rho, theta, threshold):
    lines = cv.HoughLines(image, rho, theta, threshold)
    if lines is not None:
        lines = lines.squeeze(axis=1)  # Eliminar la dimensión de 1 para las líneas
    return lines

# Función para dibujar líneas en una imagen
def draw_lines(image, lines):
    if lines is not None:
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Función de callback para los trackbars
def on_trackbar(_):
    # Detectar líneas utilizando la transformada de Hough con los parámetros actuales
    lines = detect_lines(image_gray, rho=int(cv.getTrackbarPos('rho', 'Detected Lines')),
                         theta=np.pi / 180 * cv.getTrackbarPos('theta', 'Detected Lines'),
                         threshold=cv.getTrackbarPos('threshold', 'Detected Lines'))

    # Copiar la imagen original para dibujar las líneas detectadas
    image_with_lines = np.copy(image)

    # Dibujar las líneas detectadas en la imagen copiada
    draw_lines(image_with_lines, lines)

    # Mostrar la imagen con las líneas detectadas
    cv.imshow('Detected Lines', image_with_lines)

# Cargar la imagen de prueba en escala de gris
image_path = "Imagenes_Ej/camino.tif"
image = cv.imread(image_path)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Crear una ventana para mostrar la imagen y los parámetros de la transformada de Hough
cv.namedWindow('Detected Lines')

# Crear los trackbars para ajustar los parámetros de la transformada de Hough
cv.createTrackbar('rho', 'Detected Lines', 1, 100, on_trackbar)
cv.createTrackbar('theta', 'Detected Lines', 1, 360, on_trackbar)
cv.createTrackbar('threshold', 'Detected Lines', 1, 255, on_trackbar)

# Llamar a la función de callback para inicializar la detección de líneas con los valores iniciales de los trackbars
on_trackbar(0)

# Esperar a que el usuario presione una tecla para salir
cv.waitKey(0)
cv.destroyAllWindows()
