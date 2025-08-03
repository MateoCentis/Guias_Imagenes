import SimpleITK as sitk
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#valores por ahí para arrancar:
    # time_step = 0.01 #debe ser más chico a 0.125
    # iterations = 5
    # conductance_parameter = 9.0
def smoothing(image, time_step, iterations, conductance_parameter):
    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(time_step) #(controla la cantidad de smoothing)
    smoothing.SetNumberOfIterations(iterations) #Más iteraciones -> resultados más smooths
    smoothing.SetConductanceParameter(conductance_parameter) #Más alto más smoothing (fuerza de la difusión)
    smoothing_output = smoothing.Execute(image)
    return smoothing_output

#sigma = float(0.65) #cambiar (controla suavidad del gradiente, más grande -> gradiente más suave)
def gradient(image, sigma):
    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradient_magnitude.SetSigma(sigma)
    gradient_magnitude_output = gradient_magnitude.Execute(image)
    return gradient_magnitude_output

# alpha = float(-1.3) #cambiar (pendiente de la curva)
# beta = float(average_gradient + 0.75) #cambiar (punto de inflexión de la curva)
def sigmoid(image, alpha, beta):
    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoid_output = sigmoid.Execute(image)
    return sigmoid_output

#stopping_time = float(100)#Cambiar
# seed_value = 0
def segmentacion(image, posX, posY, seed_value, stopping_time):
    fast_marching = sitk.FastMarchingImageFilter() #(la segmentación en sí)

    trial_point = (posX,posY, seed_value)

    fast_marching.AddTrialPoint(trial_point) #Agrega el punto semilla

    fast_marching.SetStoppingValue(stopping_time)#Controla que tan lejos se propaga (más grande -> más grande la región segmentada)

    fast_marching_output = fast_marching.Execute(image)

    return fast_marching_output

# time_treshold = float(100) #Cambiar
def tresholder(image, time_treshold, lower_treshold=0.0):

    tresholder = sitk.BinaryThresholdImageFilter()
    tresholder.SetLowerThreshold(lower_treshold) #Umbral por abajo
    tresholder.SetUpperThreshold(time_treshold) #Umbral superior
    tresholder.SetOutsideValue(0)
    tresholder.SetInsideValue(255)

    result = tresholder.Execute(image)
    return result

def get_click_position(image):
    global posX, posY
    posX = 0
    posY = 0
    cv.namedWindow("Click")
    def click_event(event, x, y, flags, param):
        global posX, posY
        if event == cv.EVENT_LBUTTONDOWN:
            posX = x
            posY = y
            cv.destroyWindow("Click")
    cv.setMouseCallback("Click", click_event)
    cv.imshow("Click", sitk.GetArrayFromImage(image).astype(np.uint8))
    cv.waitKey(0)
    return posX, posY

#---------------------------------------------Proceso completo-------------------------------------------------
ruta = "Imagenes_Ej/rmn.jpg"
imagen = sitk.ReadImage(ruta, sitk.sitkFloat32)

smoothing_output = smoothing(imagen, 0.01, 5, 9.0)

gradient_output = gradient(smoothing_output,0.65)

sigmoid_output = sigmoid(gradient_output, -1.3, np.mean(gradient_output)+0.75)

posX, posY = get_click_position(imagen)
segmentacion_output = segmentacion(sigmoid_output, posX, posY, 0, 100)

result = tresholder(segmentacion_output, 100.0, 0.0)

sitk.WriteImage(result,"imagen_salida.jpg")