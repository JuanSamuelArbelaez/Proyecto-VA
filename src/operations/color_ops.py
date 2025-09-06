import cv2

def obtener_tamano_tipo(img):
    return img.shape, img.dtype

def separar_canales(img):
    b, g, r = cv2.split(img)
    return r, g, b, r, g, b  # Para mantener consistencia con RGB

def a_gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def a_binaria(img, umbral=128):
    gris = a_gris(img) if len(img.shape) == 3 else img
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return binaria

def a_binaria_adaptativa(img, maxval=255, metodo=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         tipo=cv2.THRESH_BINARY, tam_bloque=11, C=2):
    gris = a_gris(img) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gris, maxval, metodo, tipo, tam_bloque, C)

def a_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
