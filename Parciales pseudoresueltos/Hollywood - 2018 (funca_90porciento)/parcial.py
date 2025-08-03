import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

#============================================================================================
def generarImagenVacia(H,W):
    return np.zeros((H,W), dtype = np.dtype('uint8'))

def datosPixel(img, x, y, r):
    imgHSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    aux = imgHSV[y-r:y+r, x-r:x+r, 0]

    media = int(np.average(aux))

    return media

def segmentarColorRGB(img, muestra, factor):
    # Muestra en BGR

    H = img.shape[0]
    W = img.shape[1]

    MediaR = np.average(muestra[:,:,2])
    MediaG = np.average(muestra[:,:,1])
    MediaB = np.average(muestra[:,:,0])

    radio = np.average(np.array([np.std(muestra[:,:,2]), np.std(muestra[:,:,1]), np.std(muestra[:,:,0])]))

    salida = generarImagenVacia(H,W)
    for x in range(0,W):
        for y in range(0,H):
            R = img[y,x,2]
            G = img[y,x,1]
            B = img[y,x,0]
            distancia = pow(pow((MediaR - R),2) + pow((MediaG - G),2) + pow((MediaB - B),2),0.5)
            #print(distancia)
            if (distancia <= factor*radio):
                salida[y,x] = 1

    return salida

def divisionesH(imagen, margen):
    H,W = imagen.shape

    suma = []
    min = 99999
    max = 0
    for y in range(0, H):
        suma.append(0)
        for x in range(0, W):
            suma[y] += imagen[y,x]
        suma[y] = suma[y] / H
        if (suma[y] > max):
            max = suma[y]
        if (suma[y] < min):
            min = suma[y]

    min = min + ((max - min) * margen / 100)

    # VERIFICO SI AL ARRANCAR A REVISAR ESTA SOBRE EL MINIMO O NO
    if (suma[0] > min):
        band = True
    else:
        band = False

    ini = 0
    divisiones = []
    for i in range(0,len(suma)):
        if (suma[i] < min): #Si la suma esta debajo del minimo
            if (band == True):
                divisiones.append([ini, i])
                band = False

        if (suma[i] > min):
            if (band == False):
                ini = i
                band = True

    if (band == True):
        divisiones.append([ini, len(suma)])

    return divisiones

def divisionesV(imagen, margen):
    H,W = imagen.shape

    suma = []
    min = 99999
    max = 0
    for x in range(0, W):
        suma.append(0)
        for y in range(0, H):
            suma[x] += imagen[y,x]
        suma[x] = suma[x] / H
        if (suma[x] > max):
            max = suma[x]
        if (suma[x] < min):
            min = suma[x]

    min = min + ((max - min) * margen / 100)

    # VERIFICO SI AL ARRANCAR A REVISAR ESTA SOBRE EL MINIMO O NO
    if (suma[0] > min):
        band = True
    else:
        band = False

    ini = 0
    divisiones = []
    for i in range(0,len(suma)):
        if (suma[i] < min): #Si la suma esta debajo del minimo
            if (band == True):
                divisiones.append([ini, i])
                band = False

        if (suma[i] > min):
            if (band == False):
                ini = i
                band = True

    if (band == True):
        divisiones.append([ini, len(suma)])

    return divisiones

def Hough(suavizada, nro):
    lines = cv.HoughLines(suavizada,1,np.pi/180,100, 0, 0)

    for rho,theta in lines[nro]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        #cv.line(aux,(x1,y1),(x2,y2),255,2)

    return np.array([[x1,y1], [x2,y2]])

def multiplicacion_mascara(imagen, mascara):
    H1 = imagen.shape[0]
    W1 = imagen.shape[1]

    H2 = mascara.shape[0]
    W2 = mascara.shape[1]

    if (H1 == H2 and W1 == W2):
        dato = imagen.dtype
        img = np.zeros((H1,W1,3), dato)
        for x in range(0,W1):
            for y in range(0,H1):
                if (mascara[y,x] == 1):
                    img[y,x] = imagen[y,x]
                else:
                    img[y,x] = [0,0,0]

        return img
    else:
        print("Las imágenes deben ser de las mismas divisiones para sumarlas")
        return 0

def Sobel(direccion):
    if (direccion == 'F'):
        return (np.array([[-1,-2,-1], [0,0,0], [1,2,1]]))
    else:
        if (direccion == 'C'):
            return (np.array([[-1,0,1], [-2,0,2], [-1,0,1]]))
        else:
            return False
#============================================================================================

img = cv.imread('./Hollywood12.jpg')
profesor = cv.imread('./tito03.jpeg')

H = img.shape[0]
W = img.shape[1]

imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# DEJO SOLO LOS NEGROS (LAS LINEAS DIVISIORIAS SON NEGRAS)
imgGray[imgGray>10] = 0

#bordes = cv.GaussianBlur(imgGray, (21,21), 0)

#BUSCO LAS LINEAS VERTICALES (LAS DEL MEDIO)
filtro = Sobel('C')
bordesv = cv.filter2D(imgGray, -1, filtro)

#BUSCO LAS LINEAS HORIZONTALES (LAS DEL MEDIO)
filtro = Sobel('F')
bordesh = cv.filter2D(imgGray, -1, filtro)

#plt.figure()
#plt.imshow(bordesv, 'gray')
#plt.show()

#plt.figure()
#plt.imshow(bordesh, 'gray')
#plt.show()

lineav2 = Hough(bordesv,0)
lineav1 = Hough(bordesv,1)

lineah2 = Hough(bordesh, 0)
lineah1 = Hough(bordesh, 3)

x1 = lineav1[0][0]
x2 = lineav2[0][0]

y1 = lineah1[0][1]
y2 = lineah2[0][1]

#OBTENGO LOS TAMAÑOS DE CADA CUADRO
tamaniox = x2 - x1
tamanioy = y2 - y1

iniy = y1-tamanioy
finy = y2+tamanioy

inix = x1-tamaniox
finx = x2+tamaniox

if (iniy < 0):
    iniy = 0
if (finy > H):
    finy = H

if (inix < 0):
    inix = 0
if (finx > W):
    finx = W
# OBTENGO LOS CUADROS
primerCuadro = imgRGB[iniy:y1, inix:x1]
cuartoCuadro = imgRGB[y1:y2, inix:x1]
septimoCuadro = imgRGB[y2:finy, inix:x1]

segundoCuadro = imgRGB[iniy:y1, x1:x2]
quintoCuadro = imgRGB[y1:y2, x1:x2]
octavoCuadro = imgRGB[y2:finy, x1:x2]

tercerCuadro = imgRGB[iniy:y1, x2:finx]
sextoCuadro = imgRGB[y1:y2, x2:finx]
novenoCuadro = imgRGB[y2:finy, x2:finx]

#ARMO ARRAY CON LOS CUADROS
cuadros = np.array([primerCuadro, segundoCuadro, tercerCuadro, cuartoCuadro, quintoCuadro, sextoCuadro, septimoCuadro, octavoCuadro, novenoCuadro])

# VERIFICO QUE CUADRO TIENE EN UNA ZONA DETERMINADA COLOR DE FONDO UNICAMENTE
for i in range(0, len(cuadros)):
    x = int(cuadros[i].shape[0]/2)
    y = int(cuadros[i].shape[1]/3)

    media = datosPixel(cuadros[i], x, y, 15)
    if (media > 106 and media < 109):
        cuadroInteres = cuadros[i]
        nro = i

#107 / 108 <- color de fondo

# SEGMENTO PROFESOR

profRGB = cv.cvtColor(profesor, cv.COLOR_BGR2RGB)

# SEGMENTO TOMANDO UN SECTOR DE LA IMAGEN DEL FONDO y OBTENGO EL "NEGATIVO"
mask = cv.GaussianBlur(segmentarColorRGB(profesor, profesor[22:27,12:21], 20), (3,3), 0)
maskneg = 1-mask

Hp = profesor.shape[0]
Wp = profesor.shape[1]

Hc = cuadroInteres.shape[0]
Wc = cuadroInteres.shape[1]

#########################################################################################
# TRATO DE BUSCAR MAS O MENOS EL CUADRO EN DONDE ESTA EL BANCO DONDE COLOCAR AL profesor

#filtro = Sobel('F')
bordes = cv.Canny(cv.cvtColor(cuadroInteres, cv.COLOR_RGB2GRAY), 100, 200)
#bordes = cv.filter2D(pepa, -1, filtro)

p = bordes.shape[0]
q = bordes.shape[1]
originall = cuadroInteres[int(p*0.2):int(p*0.8), int(q*0.2):int(q*0.8)]
auxiliar = bordes[int(p*0.2):int(p*0.8), int(q*0.2):int(q*0.8)]

#plt.figure()
#plt.imshow(auxiliar, 'gray')
#plt.show()

divisiones = divisionesH(auxiliar, 10)
divisiones1 = divisionesV(auxiliar, 5)
posy = divisiones[0][0]
posx = divisiones1[0][0]

##########################################################################################
# INICIALIZO LAS POSICIONES DE DONDE SACAR EL FRAGMENTO DONDE VOY A INSERTAR AL PROFESOR
inicy = int(p*0.2) + posy - Hp
fincy = int(p*0.2) + posy
#inicy = int(cuadroInteres.shape[1]/3)-(int(Hp/2))+5
#fincy = int(cuadroInteres.shape[1]/3)+(int(Hp/2))+1+5

inicx = int(q*0.2) + posx
fincx = int(q*0.2) + posx + Wp
#inicx = int(cuadroInteres.shape[0]/3)-int(Wp/2)+25
#fincx = int(cuadroInteres.shape[0]/3)+int(Wp/2)+1+25

aux = cuadroInteres[inicy:fincy, inicx:fincx]
#maskc = np.ones(Hc,Wc)
#maskc[int(sextoCuadro.shape[1]/3), int(sextoCuadro.shape[0]/3)]

#cv.imshow('aux', aux)
#cv.waitKey(0)

# APLICO MASCARA SOBRE EL PROFESOR Y EL FRAGMENTO DE LA IMAGEN ORIGINAL (PARA MEZCLAR CON EL BORDE)
aux = multiplicacion_mascara(aux, mask) + multiplicacion_mascara(profRGB, maskneg)

#LO INSERTO
cuadroInteres[inicy:fincy, inicx:fincx] = aux

# VERIFICACION DE QUE CUADRO ES PARA ELEGIR LA POSICION DENTRO DE LA IMAGEN ORGIINAL
if (nro == 0):
    inicioy = iniy
    finzy = y1
    iniciox = inix
    finzx = x1
else:
    if (nro == 1):
        inicioy = iniy
        finzy = y1
        iniciox = x1
        finzx = x2
    else:
        if (nro == 2):
            inicioy = iniy
            finzy = y1
            iniciox = x2
            finzx = finx
        else:
            if (nro == 3):
                inicioy = y1
                finzy = y2
                iniciox = inix
                finzx = x1
            else:
                if (nro == 4):
                    inicioy = y1
                    finzy = y2
                    iniciox = x1
                    finzx = x2
                else:
                    if (nro == 5):
                        inicioy = y1
                        finzy = y2
                        iniciox = x2
                        finzx = finx
                    else:
                        if (nro == 6):
                            inicioy = y2
                            finzy = finy
                            iniciox = inix
                            finzx = x1
                        else:
                            if (nro == 7):
                                inicioy = y2
                                finzy = finy
                                iniciox = x1
                                finzx = x2
                            else:
                                inicioy = y2
                                finzy = finy
                                iniciox = x2
                                finzx = finx

# COLOCO EL CUADRO MODIFICADO DONDE VA
imgRGB[inicioy:finzy, iniciox:finzx] = cuadroInteres

plt.figure()
plt.imshow(imgRGB)
plt.show()
