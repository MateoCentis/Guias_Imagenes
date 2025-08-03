import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import leer_imagenes_de_carpeta
from skimage.feature import graycomatrix, graycoprops

def obtener_caracteristicas(imagenes):
    distancias = [1,2,5,10] #?
    angulos = [0,np.pi/4, np.pi/2, np.pi*3/4]

    vector_caracteristicas = []
    for imagen in imagenes:
        caracteristicas = []
        
        # Información de color
        B, G, R = cv.split(imagen)
        media_rojo, std_rojo = (np.mean(R), np.std(R))
        media_verde, std_verde = (np.mean(G), np.std(G))
        media_azul, std_azul = (np.mean(B), np.std(B))
        caracteristicas.extend([media_rojo, std_rojo, media_verde, std_verde, media_azul, std_azul])
        
        # Información de textura 
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

        glcm = graycomatrix(imagen_gris, distances=distancias, angles=angulos, levels=256, symmetric=True, normed=True)
      
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        caracteristicas.extend([contrast, dissimilarity, homogeneity, energy, correlation])

        vector_caracteristicas.append(caracteristicas)  
    
    return np.array(vector_caracteristicas)

if __name__ == '__main__':
    carpeta_path_early_segmentada = "Datos/Potato/Early_Blight_segmentadas"
    # carpeta_path_early_segmentada = "Datos/Potato/Potato___Early_Blight"
    imagenes_early_segmentadas = leer_imagenes_de_carpeta(carpeta_path_early_segmentada)
    caracteristicas_early = obtener_caracteristicas(imagenes_early_segmentadas)

    carpeta_path_healthy_segmentada = "Datos/Potato/Healthy_segmentadas"
    # carpeta_path_healthy_segmentada = "Datos/Potato/Potato___Healthy"
    imagenes_healthy_segmentadas = leer_imagenes_de_carpeta(carpeta_path_healthy_segmentada)
    caracteristicas_healthy = obtener_caracteristicas(imagenes_healthy_segmentadas)
    
    carpeta_path_late_segmentada = "Datos/Potato/Late_segmentadas"
    # carpeta_path_late_segmentada = "Datos/Potato/Potato___Late_Blight"
    imagenes_late_segmentadas = leer_imagenes_de_carpeta(carpeta_path_late_segmentada)
    caracteristicas_late = obtener_caracteristicas(imagenes_late_segmentadas)
    
    np.savetxt('caracteristicas_early2.txt',caracteristicas_early)
    np.savetxt('caracteristicas_healthy2.txt',caracteristicas_healthy)
    np.savetxt('caracteristicas_late2.txt',caracteristicas_late)

