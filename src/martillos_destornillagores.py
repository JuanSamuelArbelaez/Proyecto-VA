import cv2
import numpy as np
import pandas as pd

import operations.filters as fil
import operations.color_ops as cop
import operations.morph_ops as mop
import operations.arithm_ops as aop
import operations.descriptors as desc
import operations.houg_transform as ht
import operations.hu_moments as hm
import operations.textures_first_order as tfo
import operations.textures_second_order as tso
from utils import cargar_imagen, mostrar_imagen

def analizar_martillos_destornilladores(rutas):
    for ruta in rutas:
        print(f"Analizando imagen: {ruta}")
        imagen = cargar_imagen(ruta, modo='color')

        gris = cop.a_gris(imagen)

        # Aquí iría el procesamiento específico para martillos y destornilladores
        # mostrar_imagen(imagen, "Martillos y Destornilladores")

        log = desc.laplaciano_de_gauss(imagen, (7,7))
        log = fil.filtro_blur(log)
        log = fil.filtro_sharpen(log)

        # mostrar_imagen(log, "Laplaciano de Gauss")

        _, binaria = cv2.threshold(log, 10, 170, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria_blur = fil.filtro_blur(binaria)
        binaria_sharpen = fil.filtro_sharpen(binaria_blur)


        _, binaria = cv2.threshold(binaria_sharpen, 8, 170, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # mostrar_imagen(binaria_sharpen, "Imagen Binaria después de Blur y Sharpen")

        kernel = np.ones((7, 7), np.uint8)
        binaria_dilatada = cv2.dilate(binaria, kernel, iterations=1)
        contornos, _ = cv2.findContours(binaria_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === 4. MÁSCARA DE INTERIORES ===
        mask = np.zeros_like(gris, dtype=np.uint8)
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 6000:
                continue
            cv2.drawContours(mask, [c], -1, 255, -1)

        # mostrar_imagen(mask, "Máscara de Interiores")

        mask = cv2.erode(mask, kernel, iterations=2)

        recorte_color = cv2.bitwise_and(imagen, imagen, mask=mask)

        contornos_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res = imagen.copy()
        for c in contornos_2:
            area = cv2.contourArea(c)
            if area < 6000:
                continue
            cv2.drawContours(res, [c], -1, (0, 255, 0), 3)
        
        mostrar_imagen(res, "Contornos sobre Imagen Original")

        kp_sift, des_sift, img_sift = desc.puntos_clave_sift(recorte_color)
        num_kp_sift = 0 if kp_sift is None else len(kp_sift)
        mostrar_imagen(img_sift)

        kp_orb, des_orb, img_orb = desc.puntos_clave_orb(recorte_color, nfeatures=1000)
        num_kp_orb = 0 if kp_orb is None else len(kp_orb)
        mostrar_imagen(img_orb)

        ht_circulos, circulos = ht.houg_transform_circles(recorte_color, canny_umbral=50, acumulacion_umbral=25, maxRadius=70)
        num_circulos = 0 if circulos is None else circulos.shape[1]
        mostrar_imagen(ht_circulos)
        

        




        