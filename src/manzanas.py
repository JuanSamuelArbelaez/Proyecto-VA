import cv2
import numpy as np

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


def detectar_interiores_manzana(ruta):
    """
    Detección y análisis de interiores de manzanas (mitades, rebanadas, en cascos, etc)
    
    Etapas:
        1. Preprocesamiento (grises, Laplaciano de Gauss)
        2. Binarización doble + realce de bordes
        3. Operaciones morfológicas para cerrar regiones
        4. Creación de máscara de interiores
        5. Cálculo de diferencia de canales (G - R)
        6. Detección de contornos internos
        7. Cálculo de descriptores de textura y forma
        8. Visualización de descriptores sobre la imagen
        9. Visualización y retorno de resultados
    """

    # === 1. CARGA Y PREPROCESAMIENTO ===
    img = cargar_imagen(ruta)
    gris = cop.a_gris(img)

    # Realce de bordes con Laplaciano de Gauss
    log = desc.laplaciano_de_gauss(img)
    log = cv2.medianBlur(log, 5)

    # === 2. BINARIZACIÓN DOBLE Y SUAVIZADO ===
    _, binaria = cv2.threshold(log, 8, 188, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria_blur = fil.filtro_blur(binaria)
    binaria_sharpen = fil.filtro_sharpen(binaria_blur)

    _, binaria = cv2.threshold(binaria_sharpen, 8, 188, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria_blur = fil.filtro_blur(binaria)
    binaria_sharpen = fil.filtro_sharpen(binaria_blur)

    # === 3. MORFOLOGÍA PARA CERRAR REGIONES ===
    kernel = np.ones((9, 9), np.uint8)
    binaria_dilatada = cv2.dilate(binaria_sharpen, kernel, iterations=1)
    contornos, _ = cv2.findContours(binaria_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === 4. MÁSCARA DE INTERIORES ===
    mask = np.zeros_like(gris, dtype=np.uint8)
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 6000:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    # Refinamiento morfológico
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = mop.cierre(mask, kernel)

    # === 5. APLICACIÓN DE MÁSCARA ===
    recorte_color = cv2.bitwise_and(img, img, mask=mask)
    r, g, b, _, _, _ = cop.separar_canales(recorte_color)
    diff = aop.restar(g, r)

    _, diff_binaria = cv2.threshold(diff, 80, 188, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # === 6. DETECCIÓN DE CONTORNOS INTERNOS ===
    contornos_final, _ = cv2.findContours(diff_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado_final = img.copy()

    for i, c in enumerate(contornos_final):
        area = cv2.contourArea(c)
        if area < 4000:
            continue

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0

        # Dibujar contorno y centroide
        cv2.drawContours(resultado_final, [c], -1, (0, 255, 255), 3)
        cv2.circle(resultado_final, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(resultado_final, f"Carne_{i+1}", (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # === 7. DESCRIPTORES DE TEXTURA Y FORMA ===
    tfo_media = tfo.media(diff)
    tfo_var = tfo.varianza(diff)
    tfo_std = tfo.desviacion_estandar(diff)
    tfo_entropy = tfo.entropia(diff)

    tso_contrast = tso.contraste(diff)
    tso_homog = tso.homogeneidad(diff)
    tso_dissim = tso.disimilitud(diff)
    tso_energy = tso.energia(diff)
    tso_corr = tso.correlacion(diff)
    tso_entropy = tso.entropia_glcm(diff)

    momentos_hu = hm.calcular_momentos_hu(diff)

    print("\n --- Descriptores ---")
    print(f"TFO: media={tfo_media:.3f}, var={tfo_var:.3f}, std={tfo_std:.3f}, entropía={tfo_entropy:.3f}")
    print(f"TSO: contraste={tso_contrast}, homog={tso_homog}, disim={tso_dissim}, energía={tso_energy}, corr={tso_corr}")
    print(f"Hu moments: {momentos_hu}\n")

    # === 8. VISUALIZACIÓN DE DESCRIPTORES ===
    imagen_descriptores = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
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
        cv2.putText(imagen_descriptores, t, (15, 35 + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # === 9. SALIDAS ===
    imagenes = {
        "Sharpen + Gaussiano": img,
        "Imagen a grises": gris,
        "Laplaciano de Gauss": log,
        "Binaria": binaria,
        "Bordes Sharpen": binaria_sharpen,
        "Máscara": mask,
        "Recorte color": recorte_color,
        "Resta G - R": diff,
        "Binaria de resta": diff_binaria,
        "Contornos final": resultado_final,
        "Descriptores sobre interior": imagen_descriptores,
    }

    return imagenes


def analizar_manzanas(rutas):
    """
    Ejecuta el análisis completo de una lista de imágenes de manzanas.
    """
    for ruta in rutas:
        print(("-"*50)+f"\nProcesando imagen: {ruta}\n")
        imgs = detectar_interiores_manzana(ruta)
        for nombre, im in imgs.items():
            mostrar_imagen(im, nombre)


def test1(rutas):
    for ruta in rutas:
        # === 1. Cargar y preprocesar ===
        img = cargar_imagen(ruta)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_sharp = fil.filtro_sharpen(gris)
        img_blur = fil.filtro_gaussiano(img_sharp, (5,5), 0)

        # === 2. Detección de bordes ===
        log = desc.laplaciano_de_gauss(img_blur)
        log = cv2.medianBlur(log, 5)

        # === 3. Binarización ===
        _, binaria = cv2.threshold(log, 20, 180, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # === 5. Contornos ===
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === 6. Filtrado por forma usando momentos de Hu ===
        resultado = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        interiores = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 1000:  # descartar ruido
                continue

            # calcular momentos
            M = cv2.moments(c)
            hu = cv2.HuMoments(M)
            hu_sum = np.sum(np.abs(hu))

            # las zonas interiores suelen tener contornos suaves y valores bajos de hu
            if hu_sum < 0.01:
                cv2.drawContours(resultado, [c], -1, (0,255,255), 2)
                interiores.append(c)

        # === 7. Detección de círculos internos (núcleo) con Hough ===
        circulos_img, circulos = ht.houg_transform_circles(img)

        # === 8. Descriptores (solo para análisis, no detección directa) ===
        _, _, img_sift = desc.puntos_clave_sift(img)
        _, _, img_orb = desc.puntos_clave_orb(img)

        # === 9. Salidas ===
        imagenes = {
            "Original": img,
            "Sharpen + Gaussiano": img_blur,
            "Laplaciano de Gauss": log,
            "Binaria": binaria,
            "Interiores detectados": resultado,
            "Círculos Hough": circulos_img,
            "Puntos SIFT": img_sift,
            "Puntos ORB": img_orb
        }

        for nombre, im in imagenes.items():
            mostrar_imagen(im, nombre)



def test2(rutas):
    for ruta in rutas:
        #ruta = rutas[0]
        img = cargar_imagen(ruta)
        mostrar_imagen(img, "Original")

        img_blur = fil.filtro_gaussiano(img, (5,5), 0)
        img_sharp = fil.filtro_sharpen(img_blur)
        # mostrar_imagen(img_sharp, "Sharpen")

        lap = desc.laplaciano_de_gauss(cv2.cvtColor(img_sharp, cv2.COLOR_RGB2GRAY))
        # mostrar_imagen(lap, "Laplaciano de Gauss")


        lap_neg = cv2.bitwise_not(lap)
        # mostrar_imagen(lap_neg, "Negativo Laplaciano")

        canny = fil.borde_canny(lap_neg, 10, 150)
        # mostrar_imagen(canny, "Canny")

        canny_sharp = fil.filtro_sharpen(canny, (5,5, 0))

        contornos, _ = cv2.findContours(canny_sharp , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > 2000]

        resultado = img.copy()
        for i, c in enumerate(contornos_filtrados):
            # Dibujar contorno
            cv2.drawContours(resultado, [c], -1, (0, 255, 255), 3)

            # Calcular propiedades
            area = cv2.contourArea(c)
            perimetro = cv2.arcLength(c, True)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            else:
                cx, cy = 0, 0

            # Dibujar centroide y etiqueta
            cv2.circle(resultado, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(resultado, f"A{i+1}", (cx, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        mostrar_imagen(resultado, "Contornos Detectados")
    
