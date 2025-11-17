from ultralytics import YOLO
import cv2

model = YOLO("best_1.pt") # este es el obtenido con el modelo preentrenado

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Detección", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
