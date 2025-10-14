import numpy as np

def media(img):
    return np.mean(img)

def varianza(img):
    return np.var(img)

def desviacion_estandar(img):
    return np.std(img)

def entropia(img):
    # Convertir la imagen a tipo float para evitar problemas con valores negativos
    img_float = np.float32(img) + 1e-5  # Añadir un pequeño valor para evitar log(0)
    return -np.sum(img_float * np.log(img_float))
