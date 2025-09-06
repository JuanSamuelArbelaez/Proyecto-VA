import cv2
import numpy as npl
import numpy as np

def filtro_blur(img, ksize=(5,5)):
    return cv2.blur(img, ksize)

def filtro_gaussiano(img, ksize=(5,5), sigma=0):
    return cv2.GaussianBlur(img, ksize, sigma)

def filtro_sharpen(img, kernel=None):
    if kernel is None:
        kernel = np.array([[0, -1,  0],
                           [-1,  5, -1],
                           [0, -1,  0]])
    return cv2.filter2D(img, ddepth=0, kernel=kernel)

def filtro_mediana(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def filtro_laplaciano(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

def borde_sobel(img, direccion='x', ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    if direccion=='x':
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif direccion=='y':
        return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=ksize)

def borde_prewitt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=int)
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernely)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

def borde_canny(img, umbral1=100, umbral2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return cv2.Canny(gray, umbral1, umbral2)
