# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:21:55 2021

@author: chueko
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss
import pandas as pd

#OPERACIONES SOBRE EL HISTOGRAMA

def ecualizarHistograma(img):
    imgEq=cv.equalizeHist(img)
    return imgEq

def obtenerHistograma(img):
    plt.figure()
    plt.hist(img.flatten(), 255, [0, 256])
    plt.title('histograma')
    hist1 = cv.calcHist([img], [0], None, [256], [0, 256])
    return hist1

def ecualizarDinamico(img):
## Ecualizo dinamicamente la imagen para mejorar los resultados
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    EqDinamico = cv.calcHist([cl1],[0],None,[256],[0,256])
    return EqDinamico

def ecualizaRgb(Rgb):
    Rgb[0]=cv.equalizeHist(Rgb[0])
    Rgb[1]=cv.equalizeHist(Rgb[1])
    Rgb[2]=cv.equalizeHist(Rgb[2])
    return(Rgb)
        
def ecualizaHsv(hsv):
    y=cv.split(hsv)
    y[2]=cv.equalizeHist(y[2])
    hsv=cv.merge(y)
    return(hsv)

#OPERACIONES ARITMĂTICAS

def suma(imagen1, imagen2, alfa):
    imagen1 = imagen1.astype('uint16')
    imagen2 = imagen2.astype('uint16')
    S = cv.addWeighted(imagen1, alfa, imagen2, 1 - alfa, 0.0)
    S = S.astype('uint8')
    return S


def resta(imagen1, imagen2):
    imagen1 = imagen1.astype('int16')
    imagen2 = imagen2.astype('int16')
    S = (imagen1 - imagen2)
    S = np.clip(S, 0, 255)
    S = S.astype('uint8')
    return S


def multiplica(imagen1, imagen2):
    S = (imagen1 * imagen2)
    S = S.astype('uint8')
    return S


#TRANSFORMACIONES

def lut(a,c,img):
    S = img.copy()
    S = S.astype('uint16')
    S = a * S + c
    S[S > 255] = 255
    S[S < 0] = 0
    S = S.astype('uint8')
    return S

def logaritmica(S):
    S = S.astype('uint16')
    c = 255/np.log(1+S.max()) 
    S=S+1;
    S=c*np.log(S)
    S = np.where(S < 255, S, 255)
    S = np.where(S > 0, S, 0)
    S = S.astype('uint8')
    return S

def exponencial(S,gamma):
    S=S/255
    S=np.power(S,gamma)
    return S 

#CORRECION GAMMA
def correccionGamma(img, gamma=1.0):
	invGamma = 1.0 / gamma
	tabla = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv.LUT(img, tabla)
 

#CALCULO DE ESTADĂSTICAS

def entropia(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return ss.entropy(counts, base=base)

def varianza(imagen):
    return  imagen.var()


#OPERACIONES DE UMBRALIZADO

def umbralBinario(img,umbral,maximo):
    ret, img_umbralizada = cv.threshold(img, umbral, maximo, cv.THRESH_BINARY)
    plt.figure()
    plt.subplot(121,title='imagen original')
    plt.imshow(img,'gray')
    plt.subplot(122,title='imagen umbralizada a nivel %i' %umbral)
    plt.imshow(img_umbralizada,'gray')
    return img_umbralizada

def umbralAdaptativo(img,maximo,tamanio_bloque,c):
    img_umbralizada=cv.adaptiveThreshold(img,maximo, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, tamanio_bloque, c);
    return img_umbralizada


#FILTROS

def filtroMascara(n,imagen):
    mask=np.ones((n,n),np.float32)/(n*n)
    imagen_f=cv.filter2D(imagen,-1,mask)
    imagen_f=imagen_f.astype(np.uint8)
    return imagen_f

def BoxFiltro(n,imagen):
    imagen_f=cv.boxFilter(imagen,-1,(n,n),cv.BORDER_ISOLATED )
    return imagen_f

def filtroCruz(imagen):
    mask=np.zeros((3,3),np.float32)
    mask[1,:]=1/5
    mask[:,1]=1/5
    imagen_f=cv.filter2D(imagen,-1,mask)
    imagen_f=imagen_f.astype(np.uint8)
    return imagen_f


def filtroGausiano(n,o,imagen):
    kernel = cv.getGaussianKernel(n,o)
    imagen_f=cv.filter2D(imagen,-1,kernel)
    imagen_f=imagen_f.astype(np.uint8)
    return imagen_f

def filtroMedia(n,imagen):
    imagen_f=cv.medianBlur(imagen, n)
    imagen_f=imagen_f.astype(np.uint8)
    return imagen_f

def filtroPasaAlto(suma,imagen):
    if(suma==1):
        mask=np.zeros((3,3),np.float32)
        mask[1,:]=-1
        mask[:,1]=-1
        mask[1,1]=5
    if(suma==0):
        mask=-1*(np.ones((3,3),np.float32))
        mask[1,1]=8
    if(suma==2):
        mask=1*(np.ones((3,3),np.float32))
        mask[1,:]=-2
        mask[:,1]=-2
        mask[1,1]=5
    if(suma==3):
        mask=-1*(np.ones((3,3),np.float32))
        mask[1,1]=9
    imagen_f=cv.filter2D(imagen,-1,mask)
    imagen_f=imagen_f.astype(np.uint8)
    return imagen_f

def filtroBilateral(n,m,j,imagen):
    dst = cv.bilateralFilter(imagen,n,m,j)
    return dst

def hsv2hsi(imagen):
    hsi=cv.cvtColor(imagen,cv.COLOR_RGB2HSV)
    y=cv.split(hsi)
    rgb=cv.split(imagen)
    y[2]=sum(rgb[:])/3
    y[2]=y[2].astype(np.uint8)
    hsi=cv.merge(y)
    return(hsi)            

def complementoColor(imagen):
    hsv=cv.cvtColor(imagen,cv.COLOR_RGB2HSV)
    h,s,v=cv.split(hsv)
    h=h.astype(np.uint32)
    h[:,:]=h[:,:]+90
    h[h[:,:]>180]=h[h[:,:]>180]-180
    h=h.astype(np.uint8)
    hsv=cv.merge([h,s,v])
    img=cv.cvtColor(hsv,cv.COLOR_HSV2RGB)
    return(img)


#FILTROS ESPACIALES

def obtenerLaplaciano(img):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(grayImage,cv.CV_8U)
    return laplacian

def obtenerSobelX(img,kSize):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx=cv.Sobel(grayImage,cv.CV_8U,1,0,ksize=kSize)
    return sobelx

def obtenerSobelY(img,kSize):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobely = cv.Sobel(grayImage,cv.CV_8U,0,1,ksize=kSize)
    return sobely

def obtenerPrewitt(img):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Prewitt operator
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
    
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)
    
    # Turn uint8, image fusion
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

#FILTROS COLOR

def filtraRgb(n,imagen):
    Rgb=cv.split(imagen)
    Rgb[0]=filtroPasaAlto(n,Rgb[0])
    Rgb[1]=filtroPasaAlto(n,Rgb[1])
    Rgb[2]=filtroPasaAlto(n,Rgb[2])

    Imagen_Filtrada=cv.merge(Rgb)
    return(Imagen_Filtrada)

def filtraHsv(n,imagen):
    hsv=cv.cvtColor(imagen,cv.COLOR_RGB2HSV)
    y=cv.split(hsv)
    y[2]=filtroPasaAlto(n,y[2])
    hsv_filtrado=cv.merge(y)
    return(hsv_filtrado)
    
def umbralRGB(U,imagen):
    rgb=cv.split(imagen)
    for i in range(0,3):
        rgb[i]=cv.inRange(rgb[i],U[i][0],U[i][1])
    imagen_f=cv.merge(rgb)
    return imagen_f

def umbralHSV(A,B,imagen):
    hsv=cv.cvtColor(imagen,cv.COLOR_RGB2HSV)
    imagen_hsv = cv.inRange(hsv, A,B)
    return(imagen_hsv)

#FOURIER
def fft(imagen,titulo):
    plt.figure()
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(imagen, cmap = 'gray')
    plt.title(titulo), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()   
    return fshift

def ifft(espectro):
    imagen=np.fft.fftshift(espectro)
    imagen=np.abs(np.fft.ifft2(imagen))
    # imagen=np.fft.fftshift(imagen)
    imagen=np.real(imagen)
    return imagen

img=cv.imread(r'.\imgs\1.jpg')
img.shape

#-----------------------OTRAS-----------------------
def copyTo(origen,destino,calco,mask):
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            # If the pixel of the mask is not equal to 0, then dst(x,y) = scr(x,y)
            if mask[row, col] != 0:
                 # dst_image and scr_Image must have the same number of high and wide channels, otherwise an error will be reported
                destino[row, col] = calco[row, col]  
                
            # If the pixel of the mask is equal to 0, then dst(x,y) = color original
            elif mask[row, col] == 0:
                destino[row, col] = origen[row, col]
    return destino


f=fft(cv.cvtColor(img,6),'fich')
plt.imshow(ifft(f),'gray')
# plt.imshow(obtenerLaplaciano(img),'gray')



def FileCheck(fn):
    try:
        open(fn, "r")
        return 1
    except IOError: 
        print("Error: File does not appear to exist.")
        return 0
  

imagen = cv.imread('imagenes/01.jpg')

#le mando una imagen en cualquier formato, y me dev un cuadrado
def extraerCuadradito(img, ancho , alto , centro):
    # wMax = int(ancho/2) if centro[1] + int(ancho/2) < img.shape[1] else  img.shape[1]
    # hMax = int(alto/2) if centro[0] + int(alto/2) < img.shape[0] else  img.shape[0]
    
    # wMin = int(ancho/2) if centro[1]-int(ancho/2) > 0 else  0
    # hMin = int(alto/2) if centro[0] + int(alto/2) > 0 else  0
    
    cuadradito = img[centro[0]-hMin : centro[0]+hMax , 
                     centro[1]-wMin : centro[1]+wMax ]
    return cuadradito


# Ingresa imagen SIN FORMATO
def extraerPromedioColoresHSV(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img)
    hmean = np.mean(h)
    smean = np.mean(s)
    vmean = np.mean(v)
    return hmean,smean,vmean
# print(extraerPromedioColores(imagen))        

# Ingresa imagen SIN FORMATO
def aplicarMascaraHSV(img,vMin,vMax):
    imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hsvMin = np.array(vMin,dtype=np.uint8)
    hsvMax = np.array(vMax,dtype=np.uint8)
    mask = cv.inRange(imgHSV, hsvMin, hsvMax)
    
    imagenEnmascarada = cv.bitwise_and(img, img, mask=mask)
    #Dev. la mask y la imagen final ya lista para usar
    return mask,imagenEnmascarada
# plt.imshow(aplicarMascaraHSV(imagen,[20,100,100],[40,255,255])[1])
# plt.imshow(aplicarMascaraHSV(imagen,[20,100,100],[40,255,255])[0],'gray') #CUENTA CON BLANCOs


# Le doy una mascara  y me busca la primer fila de abajo para arriba (canchas OP)
def obtenerFila(mascara):
    contador = mascara.shape[0]
    fila = 0
    tol = 10 #Cantidad de pixeles que puede estar en blanco
    for y in range(mascara.shape[0]):
        suma=0
        x=0
        #hago linea hori. abajo de todo
        # cada elem q no sea cero, no el valor de los elem
        if( np.count_nonzero(mascara[contador-1,x:mascara.shape[1]]) < tol ):
            print('estoy en la linea en y=',contador)
            #Fila buscada de la cancha <--
            fila = contador
            break
        contador=contador-1
    return fila
        
def obtenerColumna(mascara):
    inicio = 0 # Revisar esto segun el problema (seria la fila desde
               # donde arranca a buscar)
    contador = 0
    largoBuscado = 100
    columnaBuscada = 0
    tol = 15
    for y in range(mascara.shape[1]):
    #trunco en 50 xq me pisa la ultima linea
        if(np.count_nonzero( mascara [inicio:inicio+largoBuscado,contador ] ) < tol ):
            print('estoy en la linea en y=',contador)
            columnaBuscada = contador
            break
        if(contador == mascara.shape[1]-1):
            break
        contador=contador+1
    return columnaBuscada



#Funcion solo de referencia, como manipular iteracion y filtrado
# def componentesConectadasWithStats(imagen):
contador=0

imagenRGB=cv.cvtColor(imagen,cv.COLOR_BGR2RGB)
plt.figure(), plt.imshow(imagenRGB), plt.title('original')

imagenGris=cv.cvtColor(imagen,6)
#-------------------------------------------------------------------
imagenGris = filtroMedia(21,imagenGris)
ret,imagenGris=cv.threshold(imagenGris,243,255,cv.THRESH_BINARY_INV)
plt.imshow(imagenGris,'gray')
#-------------------------------------------------------------------

#componentes conectadas
ret, labels, stats, centroids=cv.connectedComponentsWithStats(imagenGris,4,cv.CV_32S)
#recorro los objetos etiquetados que encontró
#buscando cada etiqueta (LABEL) dentro de la imagen etiquetada que está en LABELS
for label in range(0,ret):
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
    (cX, cY) = centroids[label]
    
    #-------------RESTRICCIONES DE TAMAÑO-------------
    if(h > 50 and w > 50 and h < imagen.shape[0]-10 and w < imagen.shape[1]-10):
        # dibuja un círculo que envuelve al objeto
        # centro=(int(centroids[label][0]),int(centroids[label][1]))
        # imagen=cv.circle(imagen,centro,int(w/2),(255,0,0),2)
    
    # if(area>10):
        #dibuja un rectangulo 
        cv.rectangle(imagen, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        
        contador=contador+1
        #-------------PARA VER UNO A UNO LOS OBJETOS-------------
        # cv.imshow("asd",mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

print("hay", contador,"objetos")