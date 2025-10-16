import cv2
import numpy as np
import operations.filters as filters


def caracteristicas_hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(img)

    return hog_features


def puntos_clave_kaze(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(img, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return keypoints, descriptors, img_keypoints


def puntos_clave_akaze(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return keypoints, descriptors, img_keypoints

def segmentacion_grabcut(img, iteraciones=5, rect=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    mask = np.zeros(img.shape[:2], np.uint8)

    if rect is None:
        h, w = img.shape[:2]
        rect = (10, 10, w - 20, h - 20)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iteraciones, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_segmentada = img * mask2[:, :, np.newaxis]

    return img_segmentada

def laplaciano_de_gauss(img, ksize=(5,5), sigma=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = filters.filtro_gaussiano(img, ksize, sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

def flujo_optico(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape)==3 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)==3 else img2

    # Calcular el flujo óptico usando el método de Lucas-Kanade
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convertir el flujo óptico a un formato visualizable
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(img1, dtype=np.float32)

    # Apilar imágenes en escala de grises para tener 3 canales
    hsv = np.stack((hsv,hsv,hsv), axis=-1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_rgb

def puntos_clave_sift(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors, img_keypoints

def puntos_clave_orb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

    return keypoints, descriptors, img_keypoints