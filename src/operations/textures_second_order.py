import numpy as np
import cv2
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage import img_as_ubyte


def contraste(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return graycoprops(glcm, prop='contrast')

def homogeneidad(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return graycoprops(glcm, prop='homogeneity')

def disimilitud(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return graycoprops(glcm, prop='dissimilarity')

def energia(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return graycoprops(glcm, prop='energy')

def correlacion(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return graycoprops(glcm, prop='correlation')

def media_glcm(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img) 
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return np.mean(glcm)

def desviacion_estandar_glcm(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img) 
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    return np.std(glcm)

def entropia_glcm(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if len(img.shape)==3 else img
    img = img_as_ubyte(img) 
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)

    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]  # Eliminar ceros para evitar log(0)
    return -np.sum(glcm_flat * np.log(glcm_flat))