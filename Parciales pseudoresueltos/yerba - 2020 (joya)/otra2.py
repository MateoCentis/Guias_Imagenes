import cv2
import numpy as np
from utils import dibujar_contorno, trackbar_canny

# Preprocesamiento de la imagen, gaussiano + canny
def pre_proceso(image, size_kernel=5, tresh1=50, tresh2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (size_kernel, size_kernel), 0)
    bordes = cv2.Canny(blurred, tresh1, tresh2)
    return bordes

# Busca en contornos el barcode de la imagen
def find_barcode_contour(bordes):
    contours, _ = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barcode_contour = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)# Devuelve el rectángulo de área mínima que encierra el contorno, 
                                            #el objeto contiene ángulo de rotación y otras boludeces
        box = cv2.boxPoints(rect)  #Devuelve el rectángulo como una lista de 4 puntos
        box = np.intp(box) 
        width, height = rect[1]
        # Busca por relación de aspecto, aunque también podría buscarse por ancho y alto
        if width < height:
            width, height = height, width
        aspect_ratio = width / height
        if 2.0 < aspect_ratio < 5.0:  
            barcode_contour = box
            break
    return barcode_contour

# 
def warp_perspective(image, barcode_contour):
    rect = cv2.minAreaRect(barcode_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# Extrae el código de barras
def extract_barcode(warped):
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary_warped = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_warped

def procesar_imagen(ruta):
    imagen = cv2.imread(ruta)
    size_kernel = 5
    tresh1 = 0
    tresh2 = 142
    bordes = pre_proceso(imagen, size_kernel, tresh1, tresh2)
    cv2.imshow('bordes', bordes)
    cv2.waitKey(0)
    barcode_contour = find_barcode_contour(bordes)
    if barcode_contour is not None:
        warped = warp_perspective(imagen, barcode_contour)
        barcode_image = extract_barcode(warped)
        cv2.imshow('Barcode', barcode_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Barcode not found.')

procesar_imagen('yerba - 2020 (nidea)/yerba.jpg') 
