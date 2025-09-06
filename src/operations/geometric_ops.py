import cv2
import numpy as np

def cambiar_tamano(img, ancho, alto, interpolacion=cv2.INTER_LINEAR):
    return cv2.resize(img, (ancho, alto), interpolation=interpolacion)

def rotar(img, grados):
    h, w = img.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, grados, 1.0)
    return cv2.warpAffine(img, M, (w, h))
