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
import anturios as ant



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

if __name__ == "__main__":
    ruta1 = "src/data/anturio01.jpg"

    """
    pruebas_color_ops(ruta1)
    pruebas_arithm_ops(ruta1, ruta1)
    pruebas_geometric_ops(ruta1)
    pruebas_logic_ops(ruta1, ruta1)
    pruebas_morph_ops(ruta1)
    pruebas_morph_segm_ops(ruta1)
    pruebas_filters(ruta1)
    """

    rutas = ["src/data/anturio01.jpg",
             "src/data/anturio02.jpg",
             "src/data/anturio03.jpg",
             "src/data/anturio04.jpg",
             "src/data/anturio05.jpg",
             "src/data/anturio06.jpg",]
    
    for ruta in rutas:
        resultado, length = ant.detectar_anturiors(ruta)
        print("Número de anturios detectados:", length)
        mostrar_imagen(resultado,"Bordes: "+str(length))

