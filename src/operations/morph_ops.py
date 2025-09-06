import cv2
import numpy as np

# Erosión
def erosion(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.erode(img, kernel, iterations=iterations)

# Dilatación
def dilatacion(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.dilate(img, kernel, iterations=iterations)

# Apertura (Erosión seguida de Dilatación)
def apertura(img, kernel=None):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Cierre (Dilatación seguida de Erosión)
def cierre(img, kernel=None):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Gradiente morfológico (Diferencia entre Dilatación y Erosión)
def gradiente(img, kernel=None):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top-hat (diferencia entre la imagen y su apertura)
def top_hat(img, kernel=None):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black-hat (diferencia entre el cierre y la imagen)
def black_hat(img, kernel=None):
    if kernel is None:
        kernel = crear_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def crear_kernel(tamano=(5,5)):
    return np.ones(tamano, np.uint8)