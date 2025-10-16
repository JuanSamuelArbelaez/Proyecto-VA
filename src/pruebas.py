import cv2

from utils import cargar_imagen, mostrar_imagen
import operations.color_ops as cop
import operations.arithm_ops as aop
import operations.geometric_ops as gop
import operations.logic_ops as lop
import operations.morph_ops as mop
import operations.morph_segm_ops as mseg
import operations.filters as fil
import operations.descriptors as dop
import operations.textures_first_order as tfo
import operations.textures_second_order as tso
import operations.houg_transform as ht
import operations.hu_moments as hm


def pruebas_color_ops(ruta):
    img = cargar_imagen(ruta)
    mostrar_imagen(img, "Imagen Original (RGB)")

    print("Tamaño y tipo:", cop.obtener_tamano_tipo(img))

    r, g, b, _, _, _ = cop.separar_canales(img)
    mostrar_imagen(r, "Canal R", cmap="Reds")
    mostrar_imagen(g, "Canal G", cmap="Greens")
    mostrar_imagen(b, "Canal B", cmap="Blues")

    gris = cop.a_gris(img)
    mostrar_imagen(gris, "Escala de Grises")

    binaria = cop.a_binaria(img)
    mostrar_imagen(binaria, "Binaria")

    bin_adapt = cop.a_binaria_adaptativa(img)
    mostrar_imagen(bin_adapt, "Binaria Adaptativa")

    rgb = cop.a_rgb(gris)
    mostrar_imagen(rgb, "Grises a RGB")


def pruebas_arithm_ops(ruta1, ruta2):
    img1 = cargar_imagen(ruta1)
    img2 = cargar_imagen(ruta2)

    mostrar_imagen(aop.sumar(img1, img2), "Suma")
    mostrar_imagen(aop.restar(img1, img2), "Resta")
    mostrar_imagen(aop.multiplicar(img1, img2), "Multiplicación")
    mostrar_imagen(aop.dividir(img1, img2), "División")

    mostrar_imagen(aop.ajustar_brillo(img1, 50), "Brillo +50")
    mostrar_imagen(aop.ajustar_contraste(img1, 1.5), "Contraste x1.5")


def pruebas_geometric_ops(ruta):
    img = cargar_imagen(ruta)

    mostrar_imagen(gop.cambiar_tamano(img, 200, 200), "Redimensionada 200x200")
    mostrar_imagen(gop.rotar(img, 45), "Rotada 45°")


def pruebas_logic_ops(ruta1, ruta2):
    img1 = cop.a_gris(cargar_imagen(ruta1))
    img2 = cop.a_gris(cargar_imagen(ruta2))

    mostrar_imagen(lop.and_imagen(img1, img2), "AND")
    mostrar_imagen(lop.or_imagen(img1, img2), "OR")
    mostrar_imagen(lop.not_imagen(img1), "NOT")
    mostrar_imagen(lop.xor_imagen(img1, img2), "XOR")


def pruebas_morph_ops(ruta):
    img = cop.a_binaria(cargar_imagen(ruta))

    mostrar_imagen(mop.erosion(img), "Erosión")
    mostrar_imagen(mop.dilatacion(img), "Dilatación")
    mostrar_imagen(mop.apertura(img), "Apertura")
    mostrar_imagen(mop.cierre(img), "Cierre")
    mostrar_imagen(mop.gradiente(img), "Gradiente")
    mostrar_imagen(mop.top_hat(img), "Top-Hat")
    mostrar_imagen(mop.black_hat(img), "Black-Hat")


def pruebas_morph_segm_ops(ruta):
    img = cop.a_gris(cargar_imagen(ruta))
    binaria = cop.a_binaria(img)

    mostrar_imagen(mseg.eliminar_ruido_binaria(binaria), "Apertura/Cierre")
    mostrar_imagen(mseg.extraer_contornos_morfologico(binaria), "Contornos")
    mostrar_imagen(mseg.esqueletizacion(binaria), "Esqueletización")
    mostrar_imagen(mseg.rellenar_huecos(binaria), "Huecos Rellenos")
    mostrar_imagen(mseg.segmentacion_umbral(img), "Umbral Fijo")
    mostrar_imagen(mseg.segmentacion_adaptativa(img), "Umbral Adaptativo")
    mostrar_imagen(mseg.segmentacion_por_regiones(img, (10, 10)), "Por Regiones")

    # Para watershed, se requieren marcadores preparados
    _, marcadores = cv2.connectedComponents(binaria)
    mostrar_imagen(mseg.segmentacion_watershed(cargar_imagen(ruta), marcadores), "Watershed")


def pruebas_filters(ruta):
    img = cargar_imagen(ruta)

    mostrar_imagen(fil.filtro_blur(img), "Blur")
    mostrar_imagen(fil.filtro_gaussiano(img), "Gaussiano")
    mostrar_imagen(fil.filtro_sharpen(img), "Sharpen")
    mostrar_imagen(fil.filtro_mediana(img), "Mediana")
    mostrar_imagen(fil.filtro_laplaciano(img), "Laplaciano")
    mostrar_imagen(fil.borde_sobel(img), "Sobel XY")
    mostrar_imagen(fil.borde_prewitt(img), "Prewitt")
    mostrar_imagen(fil.borde_canny(img), "Canny")


def pruebas_texturas(ruta):
    img = cargar_imagen(ruta, modo='gris')

    print("=== Texturas de Primer Orden ===")
    print("Media:", tfo.media(img))
    print("Varianza:", tfo.varianza(img))
    print("Desviación estándar:", tfo.desviacion_estandar(img))
    print("Entropía:", tfo.entropia(img))

    print("\n=== Texturas de Segundo Orden (GLCM) ===")
    print("Contraste:", tso.contraste(img))
    print("Homogeneidad:", tso.homogeneidad(img))
    print("Disimilitud:", tso.disimilitud(img))
    print("Energía:", tso.energia(img))
    print("Correlación:", tso.correlacion(img))
    print("Media GLCM:", tso.media_glcm(img))
    print("Desviación estándar GLCM:", tso.desviacion_estandar_glcm(img))
    print("Entropía GLCM:", tso.entropia_glcm(img))


def pruebas_momentos_hu(ruta):
    img = cargar_imagen(ruta, modo='gris')
    hu_moments, img_proc = hm.calcular_momentos_hu(img, suavizar=True, canny=True)
    print("=== Momentos de Hu ===")
    for i, hu in enumerate(hu_moments):
        print(f"Hu[{i+1}] = {hu[0]:.5e}")
    _, img_final = hm.generar_imagen_con_momentos_completo(img_proc, hu_moments)
    mostrar_imagen(img_final, "Momentos de Hu")


def pruebas_hough(ruta):
    img = cargar_imagen(ruta, modo='gris')
    img_lineas, bordes = ht.houg_transform(img, umbral=120)
    mostrar_imagen(bordes, "Bordes (Canny)")
    mostrar_imagen(img_lineas, "Transformada de Hough (Líneas)")

    img_circulos, _ = ht.houg_transform_circles(img)
    mostrar_imagen(img_circulos, "Transformada de Hough (Círculos)")


def pruebas_descriptores(ruta1, ruta2=None):
    img = cargar_imagen(ruta1, modo='gris')

    # --- Puntos clave ---
    _, _, img_sift = dop.puntos_clave_sift(img)
    mostrar_imagen(img_sift, "Puntos Clave SIFT")

    _, _, img_orb = dop.puntos_clave_orb(img)
    mostrar_imagen(img_orb, "Puntos Clave ORB")

    _, _, img_kaze = dop.puntos_clave_kaze(img)
    mostrar_imagen(img_kaze, "Puntos Clave KAZE")

    _, _, img_akaze = dop.puntos_clave_akaze(img)
    mostrar_imagen(img_akaze, "Puntos Clave AKAZE")

    # --- Segmentación con GrabCut ---
    img_seg = dop.segmentacion_grabcut(cargar_imagen(ruta1))
    mostrar_imagen(img_seg, "Segmentación GrabCut")

    # --- Laplaciano de Gauss ---
    img_log = dop.laplaciano_de_gauss(img)
    mostrar_imagen(img_log, "Laplaciano de Gauss")

    # --- Flujo óptico si hay segunda imagen ---
    if ruta2:
        img2 = cargar_imagen(ruta2, modo='gris')
        flow_rgb = dop.flujo_optico(img, img2)
        mostrar_imagen(flow_rgb, "Flujo Óptico")


def pruebas_hog(ruta):
    img = cargar_imagen(ruta, modo='gris')
    hog_features = dop.caracteristicas_hog(img)
    print("=== Descriptores HOG ===")
    print("Vector de características:", hog_features.shape)
    
