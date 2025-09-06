import cv2

def and_imagen(img1, img2, umbral=128):
    _, bin1 = cv2.threshold(img1, umbral, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(img2, umbral, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(bin1, bin2)

def or_imagen(img1, img2, umbral=128):
    _, bin1 = cv2.threshold(img1, umbral, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(img2, umbral, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_or(bin1, bin2)

def not_imagen(img, umbral=128):
    _, bin_img = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(bin_img)

def xor_imagen(img1, img2, umbral=128):
    _, bin1 = cv2.threshold(img1, umbral, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(img2, umbral, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_xor(bin1, bin2)
