import cv2
import numpy as np
import operations.color_ops as cop
import operations.arithm_ops as aop
import operations.morph_ops as mop
import operations.morph_segm_ops as mseg
import operations.filters as fil


def detectar_anturiors(ruta):
    img = cv2.imread(ruta)

    img_shp = fil.filtro_sharpen(img.copy())

    # Separar canales
    r, g, b, _, _ ,_ = cop.separar_canales(img_shp)

    # Diferencia para resaltar el rojo
    diff = aop.restar(r, g)

    # Umbral fijo
    binaria = cop.a_binaria(diff, 80)

    # Operaciones morfológicas para limpiar
    kernel = np.ones((15,15), np.uint8)
    cerrada = mop.cierre(binaria, kernel)

    kernel = np.ones((7,7), np.uint8)
    abierta = mop.apertura(cerrada, kernel)

    # Buscar contornos
    contornos, _ = cv2.findContours(abierta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar por área para eliminar ruido
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > 1000]

    # Dibujar resultado
    resultado = cop.a_rgb(img.copy())
    cv2.drawContours(resultado, contornos_filtrados, -1, (0,255,255), 3)

    # Devolver resultados
    return resultado, len(contornos_filtrados)
