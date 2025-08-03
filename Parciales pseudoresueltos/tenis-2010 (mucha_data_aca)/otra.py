import cv2
import numpy as np

# Leer las imágenes
image = cv2.imread('image.jpg')
advertisement = cv2.imread('advertisement.png')

# Convertir las imágenes a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_advertisement = cv2.cvtColor(advertisement, cv2.COLOR_BGR2GRAY)

# Detectar puntos de interés en ambas imágenes
sift = cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray_image, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_advertisement, None)

# Emparejar los puntos de interés
bf = cv2.BFMatcher(crossCheck=True)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Filtrar los emparejamientos buenos
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Calcular la homografía
if len(good) > 7:
    src_pts = np.float32([kp.pt for kp in keypoints1 if kp.id in [m.queryIdx for m in good]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp.pt for kp in keypoints2 if kp.id in [m.trainIdx for m in good]]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Ajustar el tamaño y la posición del cartel publicitario
if M is not None:
    h, w = advertisement.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 2)

# Mostrar la imagen con el cartel publicitario ajustado
cv2.imshow('Image with adjusted advertisement', image)
cv2.waitKey(0)