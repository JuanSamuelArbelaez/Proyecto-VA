import cv2
import numpy as np
import matplotlib.pyplot as plt
import operations.filters as filters
from io import BytesIO


def calcular_momentos_hu(img, suavizar=False, canny=False, umbral=127):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Suavizado opcional
    if suavizar:
        img = filters.filtro_gaussiano(img, (5, 5), 0)

    # Umbralización a binario
    _, binary_image = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)

    # Canny opcional
    if canny:
        imagen_procesada = filters.borde_canny(binary_image, 100, 200)
    else:
        imagen_procesada = binary_image

    # Cálculo de los momentos
    moments = cv2.moments(imagen_procesada)

    # Cálculo de los momentos de Hu
    hu_moments = cv2.HuMoments(moments)

    return hu_moments, imagen_procesada


def comparar_momentos(hu1, hu2):
    """
    Calcula la distancia euclidiana entre dos conjuntos de momentos de Hu.
    """
    return np.linalg.norm(hu1 - hu2)

def generar_imagen_con_momentos_completo(img, hu_moments, title="Imagen con Momentos de Hu"):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    hu_text = "\n".join([f"Hu[{i+1}]: {hu[0]:.2e}" for i, hu in enumerate(hu_moments)])
    fig.text(0.02, 0.5, hu_text, fontsize=10, va='center')

    # Convertir a array
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_array = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)

    return fig, img_array
