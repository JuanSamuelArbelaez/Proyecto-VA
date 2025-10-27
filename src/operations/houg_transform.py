import cv2
import numpy as np
import matplotlib.pyplot as plt
import operations.filters as filters

def houg_transform(img, apertureSize=7, umbral=150):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

    img_edges = cv2.Canny(img, 50, 150, apertureSize = apertureSize)
    lines = cv2.HoughLines(img_edges, 1, np.pi / 180, umbral)

    # Si se detectan líneas, dibujarlas sobre la imagen original
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            x1 = int(rho * np.cos(theta) + 1000 * (-np.sin(theta)))
            y1 = int(rho * np.sin(theta) + 1000 * (np.cos(theta)))
            x2 = int(rho * np.cos(theta) - 1000 * (-np.sin(theta)))
            y2 = int(rho * np.sin(theta) - 1000 * (np.cos(theta)))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img, img_edges
    

def houg_transform_circles(img, dp=1.2, minDist=40, canny_umbral=100, acumulacion_umbral=30, minRadius=20, maxRadius=60):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

    gray_blur = filters.filtro_gaussiano(img, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=canny_umbral,
        param2=acumulacion_umbral,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

    return output_img, circles