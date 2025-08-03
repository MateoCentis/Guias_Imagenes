import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import utils as ut
def fft(imagen):
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return fshift,magnitude_spectrum

def detectarAnguloRot(imagen_base,imagen_rot):
    #IMAGEN TOMADA COMO REFERENCIA
    f,espectro_base=fft(cv.cvtColor(imagen_base,6))
    ret,espectro_base=cv.threshold(espectro_base,196,255,cv.THRESH_BINARY)
    lines = cv.HoughLinesP(espectro_base.astype(np.uint8), 1, np.pi/180, 180, minLineLength=280, maxLineGap=2)
    xbase1, ybase1, xbase2, ybase2 = lines[0][0]
    cv.line(espectro_base, (xbase1, ybase1), (xbase2, ybase2), (0, 255, 0), 2)
    # plt.figure(),plt.imshow(espectro_base),plt.title('sin rotar'), plt.show()
    #Calculo el vector base
    punto_base1=np.array([xbase1, ybase1])  
    punto_base2=np.array([xbase2, ybase2])
    vector_base=punto_base2-punto_base1

    #DATOS DE LA IMAGEN ROTADA
    f,espectro_rotado=fft(cv.cvtColor(imagen_rot,6))
    ret,espectro_rotado=cv.threshold(espectro_rotado,196,255,cv.THRESH_BINARY)
    lines = cv.HoughLinesP(espectro_rotado.astype(np.uint8), 1, np.pi/180, 180, minLineLength=280, maxLineGap=2)
    print(len(lines))
    x1, y1, x2, y2 = lines[0][0]
    cv.line(espectro_rotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # plt.figure(),plt.imshow(espectro_rotado),plt.title('rotado'), plt.show()
    #calculo el vector de la imagen rotada
    punto_rot1=np.array([x1, y1])
    punto_rot2=np.array([x2, y2])
    vector_rot=punto_rot2-punto_rot1

    #BUSCO EL ANGULO ENTRE EL VECTOR TOMADO COMO REFERENCIA Y EL ROTADO
    Angulo=math.acos(np.dot(vector_base, vector_rot) / (np.linalg.norm(vector_base) * np.linalg.norm(vector_rot)))
    Angulo=Angulo*180/np.pi
    return Angulo

def rotarImagen(imagen,angulo):
    h=imagen.shape[0]
    w=imagen.shape[1]
    center = (h // 2, w // 2)
    M = cv.getRotationMatrix2D(center, angulo, 1.0)
    rotated = cv.warpAffine(imagen, M, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
    return rotated


def identificarLetra(y):
    switcher = {
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'E',
        6: 'F',
        7: 'G',
        8: 'H'}
    return switcher.get(y)

def determinarGanador(puntaje_blanco,puntaje_negro,contador_rey_negro,contador_rey_blanco):
    if(puntaje_blanco>puntaje_negro and contador_rey_blanco>0):
        return 'blanco'
    elif(puntaje_blanco<puntaje_negro and contador_rey_negro>0):
        return 'oscuro'
    else:
        if(contador_rey_blanco<1):
            return('el jugador de piezas blancas ya no tiene rey')
        if(contador_rey_negro<1):
            return('el jugador de piezas oscuras ya no tiene rey')
        else:
            return 'empate'
    
def preProcesamiento(imagen_original):
    print(imagen_original.dtype)
    ut.trackbar_segmentacion_hsv(imagen_original)
    #filtrado de color
    imagenHSV=cv.cvtColor(imagen_original,cv.COLOR_BGR2HSV)
    hsvMin=np.array([0,164,0],dtype=np.uint8) #v:68
    hsvMax=np.array([255,255,255],dtype=np.uint8)# v:192
    
    mask = cv.inRange(imagenHSV, hsvMin, hsvMax)
    plt.figure(), plt.imshow(mask, cmap='gray'), plt.title('Máscara'), plt.show()
    tablero_filtrado = cv.bitwise_and(imagen_original, imagen_original, mask=mask)
    
    plt.figure(), plt.imshow(tablero_filtrado),plt.title('pre blur'), plt.show()
    
    tablero_filtrado=cv.GaussianBlur(tablero_filtrado,(7,7),0)
    
    plt.figure(), plt.imshow(tablero_filtrado),plt.title('post blur'), plt.show()
    
    #morfología para unir las partes de las piezas
    #kernel rectangular
    kernel = cv.getStructuringElement(0,(2, 6)) # (6x2) de unos
    print("KERNEL: ", kernel)
    tablero_filtrado = cv.morphologyEx(tablero_filtrado, cv.MORPH_DILATE, kernel,1)
    
    plt.figure(), plt.imshow(tablero_filtrado),plt.title('post dilate'), plt.show()
    
    #umbral binario para poder realizar la segmentación
    
    imagenGris=cv.cvtColor(tablero_filtrado,6)
    ret,imagenGris=cv.threshold(imagenGris,5,255,cv.THRESH_BINARY)
    
    plt.figure(), plt.imshow(imagenGris, cmap='gray'),plt.title('post umbral'), plt.show()
    
    return imagenGris
    

def contarPiezas(imagenGris):
    
    ret, labels, stats, centroids=cv.connectedComponentsWithStats(imagenGris,4,cv.CV_32S)
    
    h_tablero=imagenGris.shape[0]
    w_tablero=imagenGris.shape[1]
    
        #contadores de las piezas
    #------------------claras------------------
    contador_peon_blanco=0
    contador_alfil_blanco=0
    contador_torre_blanco=0
    contador_dama_blanco=0
    contador_caballo_blanco=0
    contador_rey_blanco=0
    #-----------------oscuras-------------------
    contador_peon_negro=0
    contador_alfil_negro=0
    contador_torre_negro=0
    contador_dama_negro=0
    contador_caballo_negro=0
    contador_rey_negro=0
    
    contador_negras=0
    contador_blancas=0

    
    for label in range(0,ret):

        # print('label',label)
        mask = np.array(labels, dtype=np.uint8)
        #en la máscara me quedo solo con los píxeles etiquetados con LABEL
        mask[labels == label] = 255
        mask[labels != label]= 0
        
        #puntos importantes del objeto encontrado
        x = stats[label, cv.CC_STAT_LEFT]
        y = stats[label, cv.CC_STAT_TOP]
        w = stats[label, cv.CC_STAT_WIDTH]
        h = stats[label, cv.CC_STAT_HEIGHT]
        area=stats[label,cv.CC_STAT_AREA]
        centro=(int(centroids[label][0]),int(centroids[label][1])+3)
    
        if(area>300 and w < w_tablero and h<h_tablero):
            # print('area',area)
            # cv.imshow("máscara",mask)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # print('color:',imagen_original_gris[centro[1],centro[0]])
            color=imagen_original_gris[centro[1],centro[0]]
            posx=int((centro[1]/90) + 1) # esto debe ser para encontrar la casilla
            posy=int((centro[0]/90) + 1)
            if(color<200):
                contador_negras=contador_negras+1
                # print('area', area)
                #cuento las negras
                if(area<2200):
                    print('peon oscuro en',posx,'-',identificarLetra(posy))
                    contador_peon_negro=contador_peon_negro+1
                    continue
                if(3300<=area<=3365):
                    print('caballo oscuro en',posx,'-',identificarLetra(posy))
                    contador_caballo_negro=contador_caballo_negro+1
                    continue
                if(2915<=area<=3080 and h>75):
                    print('alfil oscuro en',posx,'-',identificarLetra(posy))
                    contador_alfil_negro=contador_alfil_negro+1
                    continue
                if(2690<=area<=3030 and h<75):
                    print('torre oscura en',posx,'-',identificarLetra(posy))
                    contador_torre_negro=contador_torre_negro+1
                    continue
                if(area>3410):
                    print('rey oscuro en',posx,'-',identificarLetra(posy))
                    contador_rey_negro=contador_rey_negro+1
                    continue
                if(3366<=area<=3410):
                    print('dama oscura en',posx,'-',identificarLetra(posy))
                    contador_dama_negro=contador_dama_negro+1
                    continue
            else:
            #estoy en las blancas
                contador_blancas=contador_blancas+1
                # print('area', area)
                if(area<1900):
                    print('peon blanco en',posx,'-',identificarLetra(posy))
                    contador_peon_blanco=contador_peon_blanco+1
                    continue
                if(2610<=area<=2655 and h>73):
                    print('caballo blanco en',posx,'-',identificarLetra(posy))
                    contador_caballo_blanco=contador_caballo_blanco+1
                    continue
                if(2390<=area<=2560):
                    print('alfil blanco en',posx,'-',identificarLetra(posy))
                    contador_alfil_blanco=contador_alfil_blanco+1
                    continue
                if(2760<=area<=2790):
                    print('torre blanca en',posx,'-',identificarLetra(posy))
                    contador_torre_blanco=contador_torre_blanco+1
                    continue
                if(2800<=area<=2900):
                    print('rey blanco en',posx,'-',identificarLetra(posy))
                    contador_rey_blanco=contador_rey_blanco+1
                    continue
                if(area>3000):
                    print('dama blanca en',posx,'-',identificarLetra(posy))
                    contador_dama_blanco=contador_dama_blanco+1
                    continue
    print('cantidad de piezas oscuras:',contador_negras)
    print('cantidad de piezas claras:',contador_blancas)
    puntaje_negro=contador_peon_negro+contador_alfil_negro*3+contador_torre_negro*5+contador_dama_negro*9+contador_caballo_negro*3
    puntaje_blanco=contador_peon_blanco+contador_alfil_blanco*3+contador_torre_blanco*5+contador_dama_blanco*9+contador_caballo_blanco*3
    return puntaje_negro,puntaje_blanco,contador_rey_blanco,contador_rey_negro



#--------------PRIMERA PARTE: ROTAR SI ES NECESARIO------------------------

#utilizo la imagen de referencia para calcular la rotación
imagenSinRotar = cv.imread('B00.png')
# imagenRotada = cv.imread(r'.\imgs\B00.png')
imagenRotada=cv.imread('B01_R04.png')

angulo=detectarAnguloRot(imagenSinRotar,imagenRotada)
print('angulo:',angulo)

if(angulo<80 and angulo>3):
    rotated=rotarImagen(imagenRotada,angulo)[152:152+720,152:152+720] # Para quedar solo con el tablero
else:
    rotated=imagenRotada
    
# plt.figure(), plt.imshow(rotated), plt.title('rotada'), plt.show()

#imagen 1
imagen_original = rotated
imagen_original_gris=cv.cvtColor(imagen_original,6)

imagenGris=preProcesamiento(imagen_original)

puntaje_negro,puntaje_blanco,contador_rey_blanco,contador_rey_negro=contarPiezas(imagenGris)

print('el puntaje de las piezas blancas es',puntaje_blanco)
print('el puntaje de las piezas oscuras es',puntaje_negro)

print('el jugador con más puntaje hasta el momento es',determinarGanador(puntaje_blanco,puntaje_negro,contador_rey_negro,contador_rey_blanco))

