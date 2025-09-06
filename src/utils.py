import cv2
import matplotlib.pyplot as plt

def cargar_imagen(ruta, modo='color'):
    if modo == 'color':
        img = cv2.imread(ruta, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        # Convertir de BGR → RGB por defecto
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif modo == 'gris':
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        return img
    else:
        raise ValueError("Modo debe ser 'color' o 'gris'")

def mostrar_imagen(img, titulo="Imagen", cmap=None):
    if len(img.shape) == 2:  # Escala de grises o binaria
        plt.imshow(img, cmap='gray' if cmap is None else cmap)
    else:
        plt.imshow(img)  # Ya viene en RGB
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def guardar_imagen(ruta, img, convertir_bgr=True):
    if len(img.shape) == 3 and convertir_bgr:
        # Guardar en formato BGR (convención de OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ruta, img_bgr)
    else:
        cv2.imwrite(ruta, img)
