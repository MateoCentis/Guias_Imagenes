import numpy as np
import cv2 as cv

def rotate(img,angle):
    """Rotación de la imagen sobre el centro"""
    r = cv.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),angle,1.0)
    return cv.warpAffine(img,r,img.shape)

def detectar_rotacion(imagen, umbral1=50, umbral2=150): #esto anda pero ni idea
    edges = cv.Canny(imagen, umbral1, umbral2)
    # Detecta las líneas con la transformada de Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    mean_angle = np.mean(angles)
    return mean_angle

ruta0 = "ImgsPracticos/parrafo0.jpg"
ruta1 = "ImgsPracticos/parrafo1.jpg"

parrafo0 = cv.imread(ruta0, cv.IMREAD_GRAYSCALE)
parrafo1 = cv.imread(ruta1, cv.IMREAD_GRAYSCALE)

cv.imshow("Parrafo0", parrafo0)
cv.imshow("Parrafo1", parrafo1)

cv.waitKey(0)
cv.destroyAllWindows()

angulo0 = detectar_rotacion(parrafo0)
angulo1 = detectar_rotacion(parrafo1)

print(f"Ángulo de rotación 0: {angulo0}")
print(f"Ángulo de rotación 1: {angulo1}")

parrafo0_rotado = rotate(parrafo0,angulo0)
parrafo1_rotado = rotate(parrafo1, angulo1)

cv.imshow("parrafo0 rotado", parrafo0_rotado)
cv.imshow("parrafo1 rotado", parrafo1_rotado)

cv.waitKey(0)
