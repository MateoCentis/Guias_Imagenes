import cv2 as cv
from utils import segmentacion_hsv, ventana_trackbars
import matplotlib.pyplot as plt
import numpy as np

def segmentacion_hsv_trackbar(imagen, valores_trackbar):
    rango_hue = [valores_trackbar[0], valores_trackbar[1]]
    rango_saturation = [valores_trackbar[2], valores_trackbar[3]]
    rango_value = [valores_trackbar[4], valores_trackbar[5]]
    
    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imagen_hsv)

    mascara = np.logical_and(
        np.logical_and(rango_hue[0] <= h, h <= rango_hue[1]),
        np.logical_and(rango_saturation[0] <= s, s <= rango_saturation[1]),
        np.logical_and(rango_value[0] <= v, v <= rango_value[1])
    )
    
    mascara = np.uint8(mascara * 255)  # Convertimos la máscara a tipo uint8
    
    # Pintamos el área segmentada en rojo
    segmentacion = cv.bitwise_and(imagen, imagen, mask=mascara)
    segmentacion[np.where((segmentacion!=[0,0,0]).all(axis=2))] = [0,0,255]  # Pintamos en rojo
    
    # Convertimos la máscara a un formato donde el área segmentada sea blanca y el resto negro
    mascara_inversa = cv.bitwise_not(mascara)
    area_no_segmentada = cv.bitwise_and(imagen, imagen, mask=mascara_inversa)
    
    # Combinamos el área segmentada en rojo con el resto de la imagen
    resultado = cv.add(segmentacion, area_no_segmentada)
    
    return resultado
#-----------------------------------------1y2.Segmentación-color-------------------------------------------------
ruta = "Imagenes_Ej/rosas.jpg"
imagen = cv.imread(ruta)

sizeMascara = 10
mascara_promedio = np.ones((sizeMascara, sizeMascara), np.float32) / (sizeMascara**2)
imagen_pre = cv.filter2D(imagen, -1, mascara_promedio)

trackbar = False
if trackbar:
    variables_trackbar = ['hue0', 'hue1', 'saturation0', 'saturation1','value0','value1']
    parametros_trackbar = [[0,360],[0,360],[0,255],[0,255],[0,255],[0,255]]

    # Obtención de parámetros a través de la prueba de valores con trackbars
    ventana_trackbars(imagen_pre, 
                    variables_trackbar = variables_trackbar, 
                    parametros_trackbar = parametros_trackbar, 
                    transformacion = segmentacion_hsv_trackbar)



rango_hue = [162,191]
rango_saturacion = [35,255]

resultado, segmentacion, mascara = segmentacion_hsv(imagen_pre, rango_hue, rango_saturacion)

#---------------------------------------------3y4. Objetos-------------------------------------------------

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mascara)

#labels is a matrix the size of the input image where each element has a value equal to its label.
#Stats is a matrix of the stats that the function calculates. It has a length equal to the number of labels and a width equal to the number of stats. It can be used with the OpenCV documentation for it:
    # Statistics output for each label, including the background label, see below for available statistics. Statistics are accessed via stats[label, COLUMN] where available columns are defined below.

    # cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    # cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    # cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    # cv2.CC_STAT_HEIGHT The vertical size of the bounding box
    # cv2.CC_STAT_AREA The total area (in pixels) of the connected component

#Centroids is a matrix with the x and y locations of each centroid. 
    # The row in this matrix corresponds to the label number.

for i, (x, y, w, h, area) in enumerate(stats):
  # Skip the background label (label 0)
  if i == 0:
    continue

  # Print information about the component
  print(f"Component {i}:")
  print(f"  - Area: {area}")
#   print(f"  - Perimeter: {perimeter}") 
  print(f"  - Centroid: ({int(x + w/2)}, {int(y + h/2)})")

  # Draw a rectangle around the component (optional)
#   cv.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
  center_x = int(x + w / 2)
  center_y = int(y + h / 2)
  radius = int(np.sqrt(area / np.pi)*1.3)
  cv.circle(imagen, (center_x, center_y), radius, (0, 255, 0), 2)

# Display the image with identified components (optional)
cv.imshow("Image with Components", imagen)
cv.waitKey(0)
cv.destroyAllWindows()

#---------------------------------------------5. Opcional-------------------------------------------------

# cv.minEnclosingCircle()
# cv.minAreaRect() #Rectángulo de área mínima que encierra cada region
# cv.HuMomentos # Calcula momentos que pueden usarse
