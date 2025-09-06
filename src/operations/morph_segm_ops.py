from operations.filters import borde_canny
from operations.morph_ops import apertura, cierre
from operations.color_ops import a_gris
import cv2
import numpy as np

def eliminar_ruido_binaria(img_binaria, kernel=None, metodo='apertura'):
    if kernel is None:
        kernel = np.ones((5,5), np.uint8)
    if metodo == 'apertura':
        return apertura(img_binaria, kernel)
    elif metodo == 'cierre':
        return cierre(img_binaria, kernel)
    else:
        raise ValueError("Método debe ser 'apertura' o 'cierre'")

def extraer_contornos_morfologico(img_binaria, kernel=None):
    if kernel is None:
        kernel = np.ones((3,3), np.uint8)
    # Gradiente morfológico resalta bordes
    return cv2.morphologyEx(img_binaria, cv2.MORPH_GRADIENT, kernel)

def esqueletizacion(img_binaria):
    img_binaria = img_binaria.copy()
    skel = np.zeros(img_binaria.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img_binaria, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img_binaria, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_binaria = eroded.copy()
        if cv2.countNonZero(img_binaria) == 0:
            break
    return skel

def rellenar_huecos(img_binaria):
    # Invertir para que los huecos se conviertan en objetos
    inv_bin = cv2.bitwise_not(img_binaria)
    h, w = img_binaria.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(inv_bin, mask, (0,0), 255)
    return cv2.bitwise_or(img_binaria, cv2.bitwise_not(inv_bin))

def segmentacion_umbral(img, umbral=127, max_val=255):
    gray = a_gris(img) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, umbral, max_val, cv2.THRESH_BINARY)
    return binary

def segmentacion_adaptativa(img, max_val=255, metodo='gaussian', block_size=11, C=2):
    gray = a_gris(img) if len(img.shape) == 3 else img
    if metodo == 'gaussian':
        return cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size, C)
    elif metodo == 'mean':
        return cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block_size, C)
    else:
        raise ValueError("Metodo debe ser 'gaussian' o 'mean'")

def segmentacion_bordes(img, low_threshold=100, high_threshold=200):
    return borde_canny(img, low_threshold, high_threshold)

def segmentacion_por_regiones(img, seed_point):
    gray = a_gris(img) if len(img.shape) == 3 else img
    h, w = gray.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    img_copy = gray.copy()
    cv2.floodFill(img_copy, mask, seed_point, 255)
    return img_copy

def segmentacion_watershed(img_color, markers):
    # img_color: imagen en color BGR
    # markers: imagen de marcadores (int32), cada región con número diferente
    img_copy = img_color.copy()
    markers_result = cv2.watershed(img_copy, markers)
    img_copy[markers_result == -1] = [0,0,255]  # bordes rojos
    return img_copy
