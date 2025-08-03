import cv2
import numpy as np

def divisionesH(imagen, margen):
    H, W = imagen.shape
    suma = []
    min_val = 99999
    max_val = 0
    for y in range(H):
        suma.append(0)
        for x in range(W):
            suma[y] += imagen[y, x]
        suma[y] = suma[y] / W
        if suma[y] > max_val:
            max_val = suma[y]
        if suma[y] < min_val:
            min_val = suma[y]
    min_val = min_val + ((max_val - min_val) * margen / 100)
    band = suma[0] > min_val
    ini = 0
    divisiones = []
    for i in range(len(suma)):
        if suma[i] < min_val and band:
            divisiones.append([ini, i])
            band = False
        elif suma[i] > min_val and not band:
            ini = i
            band = True
    if band:
        divisiones.append([ini, len(suma)])
    return divisiones

def divisionesV(imagen, margen):
    H, W = imagen.shape
    suma = []
    min_val = 99999
    max_val = 0
    for x in range(W):
        suma.append(0)
        for y in range(H):
            suma[x] += imagen[y, x]
        suma[x] = suma[x] / H
        if suma[x] > max_val:
            max_val = suma[x]
        if suma[x] < min_val:
            min_val = suma[x]
    min_val = min_val + ((max_val - min_val) * margen / 100)
    band = suma[0] > min_val
    ini = 0
    divisiones = []
    for i in range(len(suma)):
        if suma[i] < min_val and band:
            divisiones.append([ini, i])
            band = False
        elif suma[i] > min_val and not band:
            ini = i
            band = True
    if band:
        divisiones.append([ini, len(suma)])
    return divisiones

def corregir_rotacion(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)
        median_angle = np.median(angles)
        (h, w) = imagen.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(imagen, M, (w, h))
        return rotated
    return imagen

def extraer_codigo_barras(imagen_path, margen=10):
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen.")
    
    imagen_rotada = corregir_rotacion(imagen)
    gray_rotada = cv2.cvtColor(imagen_rotada, cv2.COLOR_BGR2GRAY)

    divisiones_h = divisionesH(gray_rotada, margen)
    divisiones_v = divisionesV(gray_rotada, margen)

    if divisiones_h and divisiones_v:
        y1, y2 = divisiones_h[0]
        x1, x2 = divisiones_v[0]
        roi = imagen_rotada[y1:y2, x1:x2]
        return roi

    return None

# Ejemplo de uso
imagen_path = 'yerba - 2020 (nidea)/yerba.jpg'  # Reemplazar con la ruta de la imagen
roi = extraer_codigo_barras(imagen_path, 13)
if roi is not None:
    cv2.imshow('Codigo de Barras', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontró ningún código de barras.")
