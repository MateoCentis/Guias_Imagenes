import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------Construcción de EE's e imagen-------------------------------------------------
kernel0 = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
],dtype=np.uint8)
kernel1 = np.array([
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0]
],dtype=np.uint8)
kernel2 = np.array([
  [0, 1, 0],
  [0, 1, 1],
  [0, 1, 0]
],dtype=np.uint8)
kernel3 = np.array([
  [0, 1, 0],
  [0, 1, 1],
  [0, 0, 0]
],dtype=np.uint8)
kernel4 = np.array([
  [0, 1, 1],
  [0, 1, 1],
  [0, 1, 1]
],dtype=np.uint8)
kernel5 = np.array([
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 1]
],dtype=np.uint8)
imagen = np.array([
  [0, 0, 0,0,0,0,0,0,0,0,0],
  [0, 1, 1,1,0,0,0,0,0,0,0],
  [0, 0, 0,1,0,0,0,0,1,0,0],
  [0, 0, 0,1,0,0,0,1,1,1,0],
  [0, 0, 1,1,1,0,0,1,1,1,0],
  [0, 0, 0,0,0,0,0,0,0,0,0],
  [0, 1, 0,0,0,0,0,0,0,0,0],
  [0, 0, 1,0,0,0,0,1,1,1,0],
  [0, 0, 0,1,0,0,0,0,0,1,0],
  [0, 0, 0,0,0,0,1,1,1,1,0],
  [0, 1, 0,0,0,0,0,1,1,1,0],
  [0, 0, 0,0,0,0,0,0,0,0,0]
],dtype=np.uint8)
#---------------------------------------------Erosión y dilatación-------------------------------------------------
erosiones = []
dilataciones = []
kernels = [kernel0, kernel1, kernel2, kernel3, kernel4]
for i in range(len(kernels)):
    erosion = cv.erode(imagen, kernels[i], iterations=1)#Otros param. punto central y cant. iteraciones
    dilatacion = cv.dilate(imagen, kernels[i], iterations=1)
    erosiones.append(erosion)
    dilataciones.append(dilatacion)

# Crear subfiguras para mostrar resultados
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Mostrar erociones en la primera fila
for i, erosion in enumerate(erosiones):
    axes[0, i].imshow(erosion, cmap='gray')
    axes[0, i].set_title(f'Erosión {i}')
    axes[0, i].axis('on')

# Mostrar dilataciones en la segunda fila
for i, dilatacion in enumerate(dilataciones):
    axes[1, i].imshow(dilatacion, cmap='gray')
    axes[1, i].set_title(f'Dilatación {i}')
    axes[1, i].axis('on')

# Ajustar figura y mostrar
fig.suptitle('Resultados de erociones y dilataciones', fontsize=16)
plt.tight_layout()
plt.show()
#Erosiones
    #kernel0: tiene que romper todo 
    #kernel1: Deja pixel del centro en "segundo objeto" y rompe lo demás
    #kernel2: un pixel en segundo
    #kernel3: un pixel en primero, varios en el segundo, un par en el cuarto
    #kernel4: rompe todo
    #kernel5: deja primero, segundo, y cuarto
#Dilataciones
    #Agranda todo..