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
    lista_vectores = []
    lista_resultados = []

    for ruta in rutas:
        print(("-"*50)+f"\nProcesando imagen: {ruta}\n")
        imagenes, vector_caract = analizar_martillo_destornillador(ruta)
        lista_vectores.append(vector_caract)
        lista_resultados.append((ruta, imagenes))

    # Guardar CSV
    df = pd.DataFrame(lista_vectores)
    df.to_csv("resultados_mart_dest.csv", index=False)
    print("\nArchivo 'resultados_mart_dest.csv' guardado correctamente ✅")

    for resultado in lista_resultados:
        ruta, imagenes = resultado
        for nombre, img in imagenes.items():
            mostrar_imagen(img, nombre)
        
        
def analizar_martillo_destornillador(ruta):
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

    mask = np.zeros_like(gris, dtype=np.uint8)
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 6000:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    mask = cv2.erode(mask, kernel, iterations=2)

    recorte_gris = cv2.bitwise_and(gris, gris, mask=mask)

    contornos_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = imagen.copy()
    for c in contornos_2:
        area = cv2.contourArea(c)
        if area < 6000:
            continue
        cv2.drawContours(res, [c], -1, (0, 255, 0), 3)

    tfo_media = tfo.media(recorte_gris)
    tfo_var = tfo.varianza(recorte_gris)
    tfo_std = tfo.desviacion_estandar(recorte_gris)
    tfo_entropy = tfo.entropia(recorte_gris)

    tso_contrast = tso.contraste(recorte_gris)
    tso_homog = tso.homogeneidad(recorte_gris)
    tso_dissim = tso.disimilitud(recorte_gris)
    tso_energy = tso.energia(recorte_gris)
    tso_corr = tso.correlacion(recorte_gris)
    tso_entropy = tso.entropia_glcm(recorte_gris)

    momentos_hu, _ = hm.calcular_momentos_hu(recorte_gris)
    _, img_hu_m = hm.generar_imagen_con_momentos_completo(recorte_gris, momentos_hu, title="Momentos de Hu")

    # === 8. DESCRIPTORES ===
    imagen_descriptores = cv2.cvtColor(recorte_gris, cv2.COLOR_GRAY2BGR)
    overlay = imagen_descriptores.copy()

    texto = [
        f"Media: {tfo_media:.2f}",
        f"Varianza: {tfo_var:.2f}",
        f"Entropía: {tfo_entropy:.2f}",
        f"Contraste: {tso_contrast[0,0]:.2f}",
        f"Homog.: {tso_homog[0,0]:.2f}",
        f"Energía: {tso_energy[0,0]:.2f}",
        f"Correl.: {tso_corr[0,0]:.2f}"
    ]

    # Fondo negro semitransparente para legibilidad
    cv2.rectangle(overlay, (5, 5), (430, 5 + 35*len(texto)), (0, 0, 0), -1)
    imagen_descriptores = cv2.addWeighted(overlay, 0.6, imagen_descriptores, 0.4, 0)
    for i, t in enumerate(texto):
        cv2.putText(imagen_descriptores, t, (15, 35 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    kp_sift, des_sift, img_sift = desc.puntos_clave_sift(recorte_gris)
    num_kp_sift = 0 if kp_sift is None else len(kp_sift)

    kp_orb, des_orb, img_orb = desc.puntos_clave_orb(recorte_gris, nfeatures=1000)
    num_kp_orb = 0 if kp_orb is None else len(kp_orb)

    ht_circulos, circulos = ht.houg_transform_circles(recorte_gris, canny_umbral=50, acumulacion_umbral=25, maxRadius=70)
    num_circulos = 0 if circulos is None else circulos.shape[1]

    # === 9. SALIDAS ===
    vect_caracteristicas = {
        "nombre": ruta.split('/')[-1],
        "tfo_media": float(tfo_media),
        "tfo_varianza": float(tfo_var),
        "tfo_std": float(tfo_std),
        "tfo_entropia": float(tfo_entropy),
        "tso_contraste": float(tso_contrast[0,0]),
        "tso_homogeneidad": float(tso_homog[0,0]),
        "tso_disimilitud": float(tso_dissim[0,0]),
        "tso_energia": float(tso_energy[0,0]),
        "tso_correlacion": float(tso_corr[0,0]),
        "tso_entropia": float(tso_entropy),
        "hu_1": float(momentos_hu[0]),
        "hu_2": float(momentos_hu[1]),
        "hu_3": float(momentos_hu[2]),
        "hu_4": float(momentos_hu[3]),
        "hu_5": float(momentos_hu[4]),
        "hu_6": float(momentos_hu[5]),
        "hu_7": float(momentos_hu[6]),
        "num_keypoints_sift": num_kp_sift,
        "num_keypoints_orb": num_kp_orb,
        "num_circulos_hough": num_circulos
    }

    imagenes = {
        "Sharpen + Gaussiano": imagen,
        "Imagen a grises": gris,
        "Laplaciano de Gauss": log,
        "Binaria": binaria,
        "Bordes Sharpen": binaria_sharpen,
        "Máscara": mask,
        "Imagen con contornos": res,
        "Recorte gris": recorte_gris,
        "Descriptores sobre interior": imagen_descriptores,
        "Momentos de Hu": img_hu_m,
        "Puntos clave SIFT": img_sift,
        "Puntos clave ORB": img_orb,
        "Círculos Hough": ht_circulos
    }

    return imagenes, vect_caracteristicas

        




        