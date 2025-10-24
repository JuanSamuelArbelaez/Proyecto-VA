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

        _, binaria = cv2.threshold(log, 0, 210, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria_blur = fil.filtro_blur(binaria)
        binaria_sharpen = fil.filtro_sharpen(binaria_blur)


        _, binaria = cv2.threshold(binaria_sharpen, 8, 188, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # mostrar_imagen(binaria_sharpen, "Imagen Binaria después de Blur y Sharpen")

        kernel = np.ones((9, 9), np.uint8)
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
        mostrar_imagen(recorte_color, "Recorte en Escala de Grises")

        contornos_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res = imagen.copy()
        for c in contornos_2:
            area = cv2.contourArea(c)
            if area < 6000:
                continue
            cv2.drawContours(res, [c], -1, (0, 255, 0), 3)
        
        #mostrar_imagen(res, "Contornos sobre Imagen Original")
        




        