import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.ndimage import rotate

def transformada_fourier(image):
    # Realizar la Transformada de Fourier y centrar el espectro
    f_transform = fftshift(fft2(image))
    return np.abs(f_transform)

# Encuentra el ángulo de rotación haciendo varias rotaciones y usando el ángulo máximo
def find_rotation_angle(spectrum_original, spectrum_rotated):
    correlation = np.correlate(spectrum_original.flatten(), spectrum_rotated.flatten(), mode='full')
    max_index = np.argmax(correlation)
    angle = (max_index - len(spectrum_original.flatten())) * (360.0 / len(correlation))
    return angle

# Ejemplo de uso
image = plt.imread('path_to_image')  
rotated_image = rotate(image, angle=30)  

spectrum_original = transformada_fourier(image)
spectrum_rotated = transformada_fourier(rotated_image)

rotation_angle = find_rotation_angle(spectrum_original, spectrum_rotated)
print(f'Ángulo de rotación encontrado: {rotation_angle} grados')
