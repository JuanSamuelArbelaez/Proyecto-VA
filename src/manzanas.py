import cv2
import numpy as np
import matplotlib.pyplot as plt
import operations.color_ops as cop
import operations.arithm_ops as aop
import operations.morph_ops as mop
import operations.morph_segm_ops as mseg
import operations.filters as fil
import pruebas as pb
from utils import cargar_imagen, mostrar_imagen


def detectar_centros_manzana(ruta):
    # === 1. Cargar imagen ===
    img = cv2.imread(ruta)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # === 2. Preprocesamiento: filtro sharpen ===
    img_shp = fil.filtro_sharpen(img.copy())

    # === 3. Separar canales y diferenciar rojo ===
    r, g, b, _, _, _ = cop.separar_canales(img_shp)
    diff = aop.restar(r, g)

    # === 4. Umbral fijo ===
    binaria = cop.a_binaria(diff, 80)

    # === 5. Morfología ===
    kernel_cierre = np.ones((15, 15), np.uint8)
    cerrada = mop.cierre(binaria, kernel_cierre)

    kernel_apertura = np.ones((7, 7), np.uint8)
    abierta = mop.apertura(cerrada, kernel_apertura)

    # === 6. Contornos ===
    contornos, _ = cv2.findContours(abierta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > 1000]

    # === 7. Dibujar resultado y calcular propiedades ===
    resultado = cop.a_rgb(img.copy())
    propiedades = []

    for i, c in enumerate(contornos_filtrados):
        # Dibujar contorno
        cv2.drawContours(resultado, [c], -1, (0, 255, 255), 3)

        # Calcular propiedades
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0

        # Dibujar centroide y etiqueta
        cv2.circle(resultado, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(resultado, f"A{i+1}", (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Guardar propiedades
        propiedades.append({
            "id": i+1,
            "area": area,
            "perimetro": perimetro,
            "centroide": (cx, cy)
        })

    # === 8. Devolver resultados ===
    imagenes = {
        "Original": img_rgb,
        "Sharpen": cop.a_rgb(img_shp),
        "R-G": diff,
        "Binaria": binaria,
        "Post-morfología": abierta,
        "Resultado final": resultado
    }

    return imagenes, propiedades

def encontrar_caracteristicas(rutas):
    ruta1 = rutas[0]

    """
    pb.pruebas_color_ops(ruta1)
    pb.pruebas_arithm_ops(ruta1, ruta1)
    pb.pruebas_geometric_ops(ruta1)
    pb.pruebas_logic_ops(ruta1, ruta1)
    pb.pruebas_morph_ops(ruta1)
    pb.pruebas_morph_segm_ops(ruta1)
    pb.pruebas_filters(ruta1)
    """

    for ruta in rutas:
        imagenes, propiedades = detectar_anturios(ruta)
        print(f"Archivo: {ruta}")
        print("Número de anturios detectados:", len(propiedades))

        # Imprimir propiedades de cada anturio
        for p in propiedades:
            print(f"  Anturio {p['id']} -> Área: {p['area']:.2f}, "
                f"Perímetro: {p['perimetro']:.2f}, Centroide: {p['centroide']}")

        # Mostrar todas las etapas de la imagen
        for nombre, img in imagenes.items():
            mostrar_imagen(img, f"{nombre} - Detectados: {len(propiedades) if nombre=='Resultado final' else 'Procesando...'}")

        print("-" * 50)
    pb.pruebas_hough(ruta1)
    pb.pruebas_descriptores(ruta1, ruta2)
    pb.pruebas_hog(ruta1)
    pb.pruebas_texturas(ruta1)
    pb.pruebas_momentos_hu(ruta1)