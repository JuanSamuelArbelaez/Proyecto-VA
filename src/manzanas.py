import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# === Importación de operaciones ===
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
    Detección y análisis de interiores de manzanas (enteras, mitades o rebanadas)
    Incluye:
        - Filtros y realce de bordes (LoG, sharpen)
        - Binarización doble y morfología
        - Creación de máscara completa de las regiones interiores
        - Cálculo de descriptores: momentos de Hu y texturas
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

    print("\n📊 --- Descriptores ---")
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
