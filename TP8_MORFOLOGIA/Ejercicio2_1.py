#realizar apertura y cierre
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
kernel = np.ones((3,3),np.uint8)
imagen = np.zeros((9,17),np.uint8)
imagen[1,1:7] = 1
imagen[1,10:16] = 1

imagen[2,1:7] = 1
imagen[2,10:16] = 1

imagen[3,1:7] = 1
imagen[3,10:13] = 1

imagen[4, 4:13] = 1

imagen[5,1:7] = 1
imagen[5,10:13] = 1

imagen[6,1:7] = 1
imagen[6,10:13] = 1

imagen[7,1:7] = 1
imagen[7,10:16] = 1

apertura = cv.morphologyEx(imagen,cv.MORPH_OPEN,kernel)
cierre = cv.morphologyEx(imagen,cv.MORPH_CLOSE, kernel)

plt.imshow(apertura, cmap='gray')
plt.show()
plt.imshow(cierre, cmap='gray')
plt.show()

#ni idea después como usar esta poronga enorme