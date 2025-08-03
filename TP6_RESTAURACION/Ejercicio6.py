import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import ventana_trackbars, trackbar_transformacion
from icecream import ic

#---------------------------------------------PSF'S-------------------------------------------------
def calcularPSF(filter_size, radio):
    center_x = filter_size[1] // 2
    center_y = filter_size[0] // 2
    x = np.arange(0, filter_size[1], 1)
    y = np.arange(0, filter_size[0], 1)
    X, Y = np.meshgrid(x, y)
    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_image = np.zeros((filter_size[0], filter_size[1]))
    circle_image[distance_from_center <= radio] = 1
    
    return circle_image/np.sum(circle_image)
def calcular_PSF_elipse(filter_size, len, theta):
    center_x = filter_size[1] // 2
    center_y = filter_size[0] // 2
    punto_medio = (center_x, center_y)
    h = np.zeros(filter_size)
    cv.ellipse(img=h, center=punto_medio, angle=90-theta, startAngle=0, endAngle=360,axes=(0,len), color=255, thickness=cv.FILLED)
    return h/np.sum(h)
#---------------------------------------------IMPLEMENTACIONES WIENER-------------------------------------------------
#MIO
def filtro_wiener(imagen, h_PSF, NSR): #h_PSF: función de punto de esparcimiento, nsr: noise - signal - relation
    # Filtro wiener
    H = np.fft.fft2(np.fft.fftshift(h_PSF))
    H_abs = np.abs(H)
    H_abs_cuadrado = np.abs(H_abs)**2
    filtro_wiener = (1/H_abs)*(H_abs_cuadrado/(H_abs_cuadrado+NSR))
    
    # Aplicación del filtro
    imagen_fft = np.fft.fft2(imagen) #Sin hacer shift de la imagen
    imagen_filtrada = np.fft.ifftshift(imagen_fft * filtro_wiener)
    imagen_salida = np.abs(np.fft.ifft2(imagen_filtrada))

    return imagen_salida
# por qué el conjugado?
def wiener_filter(imagen, kernel, K):
    dummy = np.copy(imagen)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return np.uint8(dummy)

def wiener(input, PSF, eps, K=0.01):  # Wiener filtering，K=0.01
    input_fft = np.fft.fft2(input) #imagen_fft
    PSF_fft = np.fft.fft2(PSF) + eps #H(u,v) + epsilon
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K) # H(u,v)/( |H(u,v)|^2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1) # ifft(imagen_fft * H(u,v))
    result = np.abs(np.fft.fftshift(result))
    return result
#---------------------------------------------FIN IMPLEMENTACIONES-------------------------------------------------
##################################################################################################
#                                       IMAGEN BLURRED
##################################################################################################
primera_parte = True
if primera_parte:
    imagen = cv.imread("imagenes_varias/blur_text.jpg", cv.IMREAD_GRAYSCALE)
    imagen = imagen[:,:-1]
    shape = imagen.shape
    ic(shape)
    ver_imagenes = False
    if ver_imagenes:
        plt.imshow(imagen, cmap='gray')
        plt.show()
        imagen_roi = imagen
        plt.imshow(imagen_roi, cmap='gray')
        plt.show()

    variables_trackbar = ['R', 'SNR']
    parametros_trackbar = [[0,1000], [1,10000]]

    def transformacion(imagen, valores_trackbar):
        global shape
        radio = np.max([valores_trackbar[0] / 100,0.001])
        SNR = valores_trackbar[1]/100
        h = calcularPSF(shape, radio)
        #Diferentes tipos de wiener
        # imagen_filtrada = wiener(imagen, h,10e-3, 1/SNR)
        imagen_filtrada = filtro_wiener(imagen, h, 1/SNR)
        #wiener reemplaza las siguientes dos líneas
        # Hw = filtro_Wiener(h, 1/SNR)
        # imagen_filtrada = cv.normalize(aplicar_filtro(imagen,Hw),None,0,255,cv.NORM_MINMAX)
        return imagen_filtrada
        
    ventana_trackbars(imagen, variables_trackbar, parametros_trackbar, transformacion)

##################################################################################################
#                                       MOTION BLUR
##################################################################################################
segunda_parte = True
if segunda_parte:
    imagen_motion_blur = cv.imread("imagenes_varias/motion_blur2.jpg", cv.IMREAD_GRAYSCALE)
    shape = imagen_motion_blur.shape

    variables_trackbar = ['len', 'theta', 'snr']
    parametros_trackbar = [[100, 2500], [0,360], [1,1000]]

    def transformacion2(imagen, valores_trackbar):
        global shape
        len = np.round(valores_trackbar[0]/100).astype(np.uint8)
        theta = valores_trackbar[1]
        snr = valores_trackbar[2]/100

        h = calcular_PSF_elipse(shape, len, theta)
        # plt.imshow(h)
        # plt.show()
        imagen_salida = filtro_wiener(imagen,h,1/snr)
        # H = filtro_Wiener(h, 1/snr)
        # imagen_salida = aplicar_filtro(imagen, H)

        return imagen_salida

    trackbar_transformacion(imagen_motion_blur, variables_trackbar, parametros_trackbar, transformacion2)
