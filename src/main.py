import cv2
import numpy as np

from utils import cargar_imagen, mostrar_imagen
import operations.color_ops as cop
import operations.arithm_ops as aop
import operations.geometric_ops as gop
import operations.logic_ops as lop
import operations.morph_ops as mop
import operations.morph_segm_ops as mseg
import operations.filters as fil


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

def ruta_de_operaciones(ruta):
    # 1. Cargar imagen en RGB
    img = cargar_imagen(ruta)

    # 2. Mejorar contraste y nitidez
    img_brillo = aop.ajustar_brillo(img, 8)
    img_contraste = aop.ajustar_contraste(img_brillo, 1.1)
    img_gau = fil.filtro_gaussiano(img_contraste)
    img_nit = fil.filtro_sharpen(img_gau)

    # 3. Convertir a HSV para segmentar color morado/púrpura
    hsv = cv2.cvtColor(img_nit, cv2.COLOR_RGB2HSV)
    lower_purple = np.array([120, 40, 40])   # límites aproximados
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    mostrar_imagen(mask, "Máscara inicial (morado)")

    # 4. Limpiar máscara con morfología
    mask = mop.apertura(mask)
    mask = mop.cierre(mask)
    mostrar_imagen(mask, "Flor segmentada (limpia)")

    # 5. Transformada de distancia para separar pétalos
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = mop.dilatacion(mask, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores para watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_ws = img.copy()
    cv2.watershed(img_ws, markers)
    img_ws[markers == -1] = [255, 0, 0]  # Bordes en rojo
    mostrar_imagen(img_ws, "Segmentación con Watershed")

    # 6. Contornos para contar pétalos
    contornos, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contornos = img.copy()
    for i, c in enumerate(contornos):
        cv2.drawContours(img_contornos, [c], -1, (0, 255, 0), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contornos, str(i + 1), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    mostrar_imagen(img_contornos, f"Pétalos detectados: {len(contornos)}")
    
    print(f"🌸 Pétalos detectados: {len(contornos)}")

    return len(contornos)

if __name__ == "__main__":
    ruta1 = "src/data/flower1.png"
    ruta2 = "src/data/BlueBowl_02.jpg"

    """
    pruebas_color_ops(ruta1)
    pruebas_arithm_ops(ruta1, ruta1)
    pruebas_geometric_ops(ruta1)
    pruebas_logic_ops(ruta1, ruta1)
    pruebas_morph_ops(ruta1)
    pruebas_morph_segm_ops(ruta1)
    pruebas_filters(ruta1)
    """

    ruta_de_operaciones(ruta1)

    
