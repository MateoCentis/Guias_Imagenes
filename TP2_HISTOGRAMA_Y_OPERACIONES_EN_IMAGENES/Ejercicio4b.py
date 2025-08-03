import cv2 as cv

#CLAHE: Técnica avanzada de procesamiento de imágenes que mejora el contraste local de una imagen
    # Mejora el contraste local sin afectar negativamente otras partes de la imagen.
    # Evita la sobreamplificación del ruido.
    # Preserva los detalles de la imagen.
# Carga una imagen en escala de grises
imagen = cv.imread("Imagenes_Ej/imagenD.tif", cv.IMREAD_GRAYSCALE)

# Aplica CLAHE
# Limita el contraste en cada mosaico utilizando un parámetro llamado “clip limit”
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_clahe = clahe.apply(imagen)

# Muestra las imágenes original y mejorada
cv.imshow("Imagen Original", imagen)
cv.imshow("Imagen con CLAHE", imagen_clahe)
cv.waitKey(0)
cv.destroyAllWindows()
