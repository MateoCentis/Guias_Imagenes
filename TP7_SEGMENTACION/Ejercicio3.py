import SimpleITK as sitk
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np #Ver segmentacion.py ahí está más prolijo
#---------------------------------------------Parámetros-------------------------------------------------
ruta = "Imagenes_Ej/rmn.jpg"

# plt.imshow(imagen)
# plt.show()
imagen = sitk.ReadImage(ruta, sitk.sitkFloat32)
height = imagen.GetHeight()
width = imagen.GetWidth()
#---------------------------------------------Smoothing-------------------------------------------------
#armado de función
time_step = 0.01 #debe ser más chico a 0.125
iterations = 5
conductance_parameter = 9.0
smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
smoothing.SetTimeStep(time_step) #(controla la cantidad de smoothing)
smoothing.SetNumberOfIterations(iterations) #Más iteraciones -> resultados más smooths
smoothing.SetConductanceParameter(conductance_parameter) #Más alto más smoothing (fuerza de la difusión)
smoothing_output = smoothing.Execute(imagen)

# Visualización intermedia
# image_data = np.reshape(smoothing_output, (height, width))
# plt.imshow(image_data, cmap='gray')
# plt.show()

#---------------------------------------------Gradient-------------------------------------------------
sigma = float(0.65) #cambiar (controla suavidad del gradiente, más grande -> gradiente más suave)
gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
gradient_magnitude.SetSigma(sigma)
gradient_magnitude_output = gradient_magnitude.Execute(smoothing_output)

average_gradient = np.mean(gradient_magnitude_output)
print(average_gradient)
print(np.min(gradient_magnitude_output))
print(np.max(gradient_magnitude_output))
# Visualización intermedia
# image_data = np.reshape(gradient_magnitude_output, (height, width))
# plt.imshow(image_data, cmap='gray')
# plt.show()
#---------------------------------------------Sigmoid-------------------------------------------------
alpha = float(-1.3) #cambiar (pendiente de la curva)
beta = float(average_gradient + 0.75) #cambiar (punto de inflexión de la curva)
sigmoid = sitk.SigmoidImageFilter()
sigmoid.SetOutputMinimum(0.0)
sigmoid.SetOutputMaximum(1.0)
sigmoid.SetAlpha(alpha)
sigmoid.SetBeta(beta)
sigmoid_output = sigmoid.Execute(gradient_magnitude_output)

# image_data = np.reshape(sigmoid_output, (height, width))
# plt.imshow(image_data, cmap='gray')
# plt.show()
#---------------------------------------------Fast-Marching-------------------------------------------------
stopping_time = float(100)#Cambiar
posicionX = 149
posicionY = 143
seed_position = (posicionX,posicionY)
seed_value = 0
fast_marching = sitk.FastMarchingImageFilter() #(la segmentación en sí)

trial_point = (seed_position[0],seed_position[1], seed_value)

fast_marching.AddTrialPoint(trial_point) #Agrega el punto semilla

fast_marching.SetStoppingValue(stopping_time)#Controla que tan lejos se propaga (más grande -> más grande la región segmentada)

fast_marching_output = fast_marching.Execute(sigmoid_output)

#---------------------------------------------Tresholder-------------------------------------------------
time_treshold = float(100) #Cambiar
tresholder = sitk.BinaryThresholdImageFilter()
tresholder.SetLowerThreshold(0.0) #Umbral por abajo
tresholder.SetUpperThreshold(time_treshold) #Umbral superior
tresholder.SetOutsideValue(0)
tresholder.SetInsideValue(255)

result = tresholder.Execute(fast_marching_output)

sitk.WriteImage(result,"imagen_salida.jpg")