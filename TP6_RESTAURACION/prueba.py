import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2


# Simulated motion blur
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # Normalize the luminance of the point spread function
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# Blur the image with motion
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # Take the Fourier transform of a two-dimensional array
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(input, PSF, eps):  # Inverse filtering
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # Noise power, that's given，consider epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # Compute the inverse Fourier transform of F(u,v)
    result = np.abs(fft.fftshift(result))
    return result


def wiener(input, PSF, eps, K=0.01):  # Wiener filtering，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


image = cv2.imread('imagenes_varias/motion_blur.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
img_h = image.shape[0]
img_w = image.shape[1]
plt.figure(1)
plt.xlabel("Original Image")
plt.gray()
plt.imshow(image)  # Show the original image
plt.figure(2)
plt.gray()
# Motion blur
PSF = motion_process((img_h, img_w), 60)
blurred = np.abs(make_blurred(image, PSF, 1e-3))

plt.subplot(231)
plt.xlabel("Motion blurred")
plt.imshow(blurred)

result = inverse(blurred, PSF, 1e-3)  # Inverse filtering
plt.subplot(232)
plt.xlabel("inverse deblurred")
plt.imshow(result)

result = wiener(blurred, PSF, 1e-3)  # Wiener filtering
plt.subplot(233)
plt.xlabel("wiener deblurred(k=0.01)")
plt.imshow(result)

blurred_noisy = blurred + 0.1 * blurred.std() * \
                np.random.standard_normal(blurred.shape)  # Add noise,standard_normal is Generating random functions

plt.subplot(234)
plt.xlabel("motion & noisy blurred")
plt.imshow(blurred_noisy)  # Displays images with added noise and motion blur

result = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  # The image with added noise is inversely filtered
plt.subplot(235)
plt.xlabel("inverse deblurred")
plt.imshow(result)

result = wiener(blurred_noisy, PSF, 0.1 + 1e-3)  # Wiener filtering is performed on the image with added noise
plt.subplot(236)
plt.xlabel("wiener deblurred(k=0.01)")
plt.imshow(result)

plt.show()







# Aplicación en video
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cap = cv.VideoCapture('vtest.avi')
# create a list of first 5 frames
img = [cap.read()[1] for i in range(5)]
# convert all to grayscale
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
# convert all to float64
gray = [np.float64(i) for i in gray]
# create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10
# Add this noise to images
noisy = [i+noise for i in gray]
# Convert back to uint8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
# Denoise 3rd frame considering all the 5 frames
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()
"""