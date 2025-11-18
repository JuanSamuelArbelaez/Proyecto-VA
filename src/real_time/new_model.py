from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best_new_model_2.pt") # este es el obtenido entrenado desde cero

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_bl = cv2.GaussianBlur(frame, (5,5), 0)

    kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]])
    
    frame_sh =  cv2.filter2D(frame_bl, ddepth=0, kernel=kernel)

    results = model(frame_sh)
    annotated = results[0].plot()

    cv2.imshow("Detección", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
