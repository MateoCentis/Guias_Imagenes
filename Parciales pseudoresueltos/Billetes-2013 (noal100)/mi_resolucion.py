import cv2 as cv
import numpy as np
import utils as ut

#############################################################################################
#                              Carga de imágenes                                        
#############################################################################################
dos_pesos_1 = cv.imread("Billetes_Generico/2_1.jpg")
dos_pesos_2 = cv.imread("Billetes_Generico/2_2.jpg")
dos_pesos_3 = cv.imread("Billetes_Generico/2_3.jpg")
dos_pesos_4 = cv.imread("Billetes_Generico/2_4.jpg")

billetes_dos_pesos = [dos_pesos_1, dos_pesos_2, dos_pesos_3, dos_pesos_4]

cinco_pesos_1 = cv.imread("Billetes_Generico/5_1.jpg")
cinco_pesos_2 = cv.imread("Billetes_Generico/5_2.jpg")
cinco_pesos_3 = cv.imread("Billetes_Generico/5_3.jpg")
cinco_pesos_4 = cv.imread("Billetes_Generico/5_4.jpg")

billetes_cinco_pesos = [cinco_pesos_1, cinco_pesos_2, cinco_pesos_3, cinco_pesos_4]

diez_pesos_1 = cv.imread("Billetes_Generico/10_1.jpg")
diez_pesos_2 = cv.imread("Billetes_Generico/10_2.jpg")
diez_pesos_3 = cv.imread("Billetes_Generico/10_3.jpg")
diez_pesos_4 = cv.imread("Billetes_Generico/10_4.jpg")

billetes_diez_pesos = [cinco_pesos_1, cinco_pesos_2, cinco_pesos_3, cinco_pesos_4]

veinte_pesos_1 = cv.imread("Billetes_Generico/20_1.jpg")
veinte_pesos_2 = cv.imread("Billetes_Generico/20_2.jpg")
veinte_pesos_3 = cv.imread("Billetes_Generico/20_3.jpg")
veinte_pesos_4 = cv.imread("Billetes_Generico/20_4.jpg")

billetes_veinte_pesos = [veinte_pesos_1, veinte_pesos_2, veinte_pesos_3, veinte_pesos_4]

cincuenta_pesos_1 = cv.imread("Billetes_Generico/50_1.jpg")
cincuenta_pesos_2 = cv.imread("Billetes_Generico/50_2.jpg")
cincuenta_pesos_3 = cv.imread("Billetes_Generico/50_3.jpg")
cincuenta_pesos_4 = cv.imread("Billetes_Generico/50_4.jpg")

billetes_cincuenta_pesos = [cincuenta_pesos_1, cincuenta_pesos_2, cincuenta_pesos_3, cincuenta_pesos_4]

cien_pesos_1 = cv.imread("Billetes_Generico/100_1.jpg")
cien_pesos_2 = cv.imread("Billetes_Generico/100_2.jpg")
cien_pesos_3 = cv.imread("Billetes_Generico/100_3.jpg")
cien_pesos_4 = cv.imread("Billetes_Generico/100_4.jpg")

billetes_cien_pesos = [cien_pesos_1, cien_pesos_2, cien_pesos_3, cien_pesos_4]

#############################################################################################
#                                Ver que tipo de billete es                                        
#############################################################################################

def tipo_billete(imagen):
    _ , mascara_dos_pesos = ut.segmentacion_hsv(imagen, [103,117], [80,150])#14 y 
    _ , mascara_cinco_pesos = ut.segmentacion_hsv(imagen, [40,54], [45,115])
    _ , mascara_diez_pesos = ut.segmentacion_hsv(imagen, [2,13], [80,150])
    _ , mascara_veinte_pesos = ut.segmentacion_hsv(imagen, [0,10], [110,160])
    _ , mascara_cincuenta_pesos = ut.segmentacion_hsv(imagen, [138,152], [0,70])
    _ , mascara_cien_pesos = ut.segmentacion_hsv(imagen, [128,140], [31,133])

    area_dos_pesos = ut.calcular_area_imagen_binaria(mascara_dos_pesos)
    area_cinco_pesos = ut.calcular_area_imagen_binaria(mascara_cinco_pesos)
    area_diez_pesos = ut.calcular_area_imagen_binaria(mascara_diez_pesos)
    area_veinte_pesos = ut.calcular_area_imagen_binaria(mascara_veinte_pesos)
    area_cincuenta_pesos = ut.calcular_area_imagen_binaria(mascara_cincuenta_pesos)
    area_cien_pesos = ut.calcular_area_imagen_binaria(mascara_cien_pesos)

    areas = [area_dos_pesos, area_cinco_pesos, area_diez_pesos, area_veinte_pesos, area_cincuenta_pesos, area_cien_pesos]
    print(areas)
    #retornar el area mayor
    indice = np.argmax(areas)

    if indice == 0:
        return "Billete de dos pesos"
    elif indice == 1:
        return "Billete de 5 pesos"
    elif indice == 2:
        return "Billete de 10 pesos"
    elif indice == 3:
        return "Billete de 20 pesos"
    elif indice == 4:
        return "Billete de 50 pesos"
    else:
        return "Billete de 100 pesos"

# ut.trackbar_segmentacion_hsv(diez_pesos_2)
for billete in billetes_diez_pesos:
    print("--------------------------------")
    tipo = tipo_billete(billete)
    print(tipo)