import cv2 as cv
import numpy as np
import utils as ut
def trackbar_hough_lineas(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        tresh_canny = valores_trackbar[0]
        tresh_hough = valores_trackbar[1]
        min_theta = valores_trackbar[2]
        max_theta = valores_trackbar[3]
        if min_theta > max_theta:
            min_theta, max_theta = max_theta, min_theta
        bordes = cv.Canny(imagen, tresh_canny, 255)
        lineas = cv.HoughLines(bordes, 1, np.pi/180, tresh_hough, min_theta=min_theta, max_theta=max_theta)
        
        if lineas is not None:
            for line in lineas:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(imagen_salida, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return imagen_salida

    variables_trackbar = ['tresh_canny', 'acumulador_hough','min_theta','max_theta']
    parametros_trackbar = [[0, 255],[0, 255], [0,360],[0, 360]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)
def trackbar_hough_lineas_and_canny(imagen):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        tresh_canny = valores_trackbar[0]
        tresh_hough = valores_trackbar[1]
        min_theta = valores_trackbar[2]
        max_theta = valores_trackbar[3]

        if min_theta > max_theta:
            min_theta, max_theta = max_theta, min_theta

        bordes = cv.Canny(imagen, tresh_canny, 255)
        
        lines = cv.HoughLines(bordes, 1, np.pi / 180, tresh_hough, min_theta=min_theta, max_theta=max_theta)
        
        line_img = np.zeros_like(bordes)
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(line_img, (x1, y1), (x2, y2), 255, 2)  # Draw lines in white color
        
        line_img = cv.resize(line_img, (imagen.shape[1], imagen.shape[0]))
        combined = cv.bitwise_and(line_img, bordes)

        imagen_salida[combined > 0] = [0, 0, 255]

        return imagen_salida

    variables_trackbar = ['tresh_canny', 'acumulador_hough','min_theta','max_theta']
    parametros_trackbar = [[0, 255],[0, 255], [0,360],[0, 360]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)

def get_border_type(tipo_borde):
    switcher = {
        0: cv.BORDER_CONSTANT,# Pads the image with a constant value (specified by borderValue).
        1: cv.BORDER_REPLICATE,#Repeats the border pixels.
        2: cv.BORDER_REFLECT,# Reflects the border pixels. For example, fedcba|abcdefgh|hgfedcb.
        3: cv.BORDER_WRAP,#Wraps around the image. For example, cdefgh|abcdefgh|abcdefg.
        4: cv.BORDER_REFLECT_101,  # Also cv.BORDER_DEFAULT,  Reflects the border pixels but the border pixel itself is not reflected. For example, gfedcb|abcdefgh|gfedcba.
        5: cv.BORDER_TRANSPARENT,#The pixels beyond the image are not modified.
        6: cv.BORDER_ISOLATED,#Treats all border pixels as isolated pixels. It has no padding and hence is used when no border is needed.
    }
    return switcher.get(tipo_borde, cv.BORDER_CONSTANT)
def trackbar_erosion(imagen,kernel):
    def transformacion(imagen, valores_trackbar):
        imagen_salida = imagen.copy()
        iteraciones = valores_trackbar[0]
        tipo_borde = valores_trackbar[1]
        clave_tipo = get_border_type(tipo_borde)
        valor_borde = valores_trackbar[2]
        imagen_salida = cv.erode(imagen_salida, kernel, iterations=iteraciones, 
                                 borderType=clave_tipo, borderValue=valor_borde)
        return imagen_salida
    
    variables_trackbar = ['iteraciones','tipo_borde','valor_borde']
    parametros_trackbar = [[1,50],[0,6],[0,1]]

    trackbar_transformacion_val(imagen, 
                            variables_trackbar=variables_trackbar, 
                            parametros_trackbar=parametros_trackbar, 
                            transformacion=transformacion)
    
def trackbar_transformacion_val(imagen, variables_trackbar, parametros_trackbar, transformacion):
    def on_trackbar_change(val=None):
        valores_trackbar = [cv.getTrackbarPos(var, 'Trackbars') for var in variables_trackbar]
        imagen_transformada = transformacion(imagen, valores_trackbar)
        cv.imshow('Transformacion', imagen_transformada)

    cv.namedWindow('Trackbars')
    for var, (min_val, max_val) in zip(variables_trackbar, parametros_trackbar):
        cv.createTrackbar(var, 'Trackbars', min_val, max_val, on_trackbar_change)

    on_trackbar_change()
    cv.waitKey(0)
    cv.destroyAllWindows()

yerba = cv.imread("yerba.jpg")
yerba_rotada = cv.imread("yerba_rotada.jpg")

# ut.trackbar_angulo_rotacion(yerba, yerba_rotada)
ut.trackbar_angulo_rotacion_una_imagen(yerba_rotada)
