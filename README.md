# Proyecto: Detección de Martillos y Destornilladores

Resumen
-------
Repositorio para experimentos de visión por computador orientados a la detección y extracción de martillos y destornilladores. Incluye:
- Scripts de pruebas y preprocesamiento (extracción de máscaras y recortes).
- Ejecución en tiempo real usando modelos YOLO (pre_entrenado y nuevo).

Ejecución
---------
- El script principal (main) se ejecuta desde la raíz del proyecto (la carpeta que contiene `src/`):
  - `python src/main.py`
  - Esto invoca `martillos_destornilladores.analizar_martillos_destornilladores` sobre rutas definidas en `src/main.py`.
- Los scripts de tiempo real deben ejecutarse desde su carpeta `src/real_time` (para respetar rutas y cargas relativas del entorno):
  - Ir a la carpeta: `cd src/real_time`
  - Ejecutar:
    - Modelo pre-entrenado: `python pre_trained.py`
    - Modelo nuevo: `python new_model.py`

Notas:
- Ignorar los módulos de manzanas y anturios para este proyecto; están presentes pero no forman parte del flujo principal descrito aquí.
- Ejecutar `python src/main.py` desde la raíz del proyecto; ejecutar los scripts de tiempo real desde `src/real_time`.

Dependencias
------------
- Python 3.8+
- OpenCV: `pip install opencv-python`
- Ultraytics YOLO: `pip install ultralytics`
- NumPy, Pandas: `pip install numpy pandas`
- Otros módulos usados en `src/operations` (no listados como paquete pip)

Preprocesamiento y razonamiento (función analizar_martillos_destornilladores)
-----------------------------------------------------------------------------
La función `analizar_martillos_destornilladores` sigue este flujo para obtener máscaras y recortes de las herramientas en imágenes controladas:

1. Carga en color y conversión a gris para cómputos por niveles de intensidad.
2. Detección de bordes y realce:
   - Aplicación de Laplaciano de Gauss (LoG) para resaltar bordes y contornos relevantes.
   - Filtros de blur y sharpen para reducir ruido y luego mejorar bordes sin crear artefactos excesivos.
3. Umbral adaptativo + Otsu para binarizar la imagen resultante y separar primer plano/fondo.
4. Operaciones morfológicas (dilatación y erosión) con un kernel grande para consolidar regiones y eliminar pequeños ruidos.
5. Detección de contornos y filtrado por área mínima (p. ej. 6000 px) para descartar objetos pequeños o ruido.
6. Construcción de una máscara rellenando los contornos que cumplen el área mínima y refinamiento mediante erosión.
7. Aplicación de la máscara sobre la imagen en color para obtener el recorte (region-of-interest) de las herramientas.
8. Sobre el recorte se calculan descriptores de textura (primer y segundo orden), momentos de Hu y puntos clave (SIFT/ORB) y transformada de Hough para círculos. Estos vectores de características se guardan en CSV para análisis posterior.

Con este workflow se logra en imágenes controladas:
- Crear máscaras sólidas y rellenas de las herramientas (no solo contornos), lo que permite recortes limpios.
- Reducir falsos positivos filtrando por área y aplicando procesamiento morfológico.
- Obtener vectores robustos para clasificación/analítica posterior.

Librerías propias y organización
-------------------------------
El proyecto usa implementaciones locales para encapsular operaciones concretas con OpenCV:
- `src/operations/color_ops.py`        : conversiones de color (a gris, etc.)
- `src/operations/descriptors.py`      : SIFT, ORB, LoG y funciones de dibujo/visualización
- `src/operations/filters.py`          : blur, sharpen y otros kernels personalizados
- `src/operations/houg_transform.py`   : envoltura para transformadas de Hough
- `src/operations/hu_moments.py`       : cálculo y normalización de momentos de Hu
- `src/operations/textures_first_order.py`  : funciones de media, varianza, entropía
- `src/operations/textures_second_order.py` : GLCM: contraste, homogeneidad, energía, correlación, disimilitud, entropía
- `src/utils.py`                       : utilidades para carga/visualización de imágenes

Tiempo real con YOLO (filtros aplicados y rendimiento)
------------------------------------------------------
- En los scripts de tiempo real (`src/real_time/pre_trained.py` y `src/real_time/new_model.py`) se aplicaron únicamente:
  - Blur Gaussiano (GaussianBlur) para suavizar ruido.
  - Un filtro sharpen (convolution 3x3) para recuperar nitidez.
- Objetivo: mejorar levemente la calidad visual sin añadir latencia significativa ni alterar el comportamiento del detector. Las entradas al modelo siguen siendo imágenes “normales” del dataset (no máscaras ni contornos), por lo que el modelo opera en el mismo dominio de datos que en entrenamiento.

Rendimiento de modelos
----------------------
- Modelo pre_entrenado (`pre_trained`): accuracy reportado ~ 84%
- Modelo nuevo (`new_model` entrenado desde cero): accuracy reportado ~ 29.7%

Notas finales
-------------
- Los CSV de características se guardan desde la función de análisis (`resultados_martillos_destornilladores.csv`) en la raíz de ejecución.
