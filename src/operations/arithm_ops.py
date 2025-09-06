import cv2
import numpy as np

def sumar(img1, img2):
    return cv2.add(img1, img2)

def restar(img1, img2):
    return cv2.subtract(img1, img2)

def multiplicar(img1, img2):
    return cv2.multiply(img1, img2)

def dividir(img1, img2):
    return cv2.divide(img1, img2)

def ajustar_brillo(img, valor=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + valor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def ajustar_contraste(img, alfa=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * alfa, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

