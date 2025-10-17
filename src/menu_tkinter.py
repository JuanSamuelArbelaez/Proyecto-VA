import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

import cv2
import io

def lanzar_menu_interactivo(lista_resultados, df):
    """
    Menú interactivo con tabs:
    1. Visualización por imagen.
    2. Comparación de vectores.
    """

    ventana = tk.Tk()
    ventana.title("Análisis de Manzanas - Visión Artificial")
    ventana.geometry("1000x850")
    ventana.configure(bg="#1e1e1e")

    # Crear tabs
    tabs = ttk.Notebook(ventana)
    tab_visual = tk.Frame(tabs, bg="#1e1e1e")
    tab_comp = tk.Frame(tabs, bg="#1e1e1e")

    tabs.add(tab_visual, text="Resultados por Imagen")
    tabs.add(tab_comp, text="Comparación entre Imágenes")
    tabs.pack(expand=True, fill="both")

    # ------------------------------
    # 🖼️ TAB 1: VISUALIZACIÓN DE RESULTADOS
    # ------------------------------
    index = tk.IntVar(value=0)

    frame_controls = tk.Frame(tab_visual, bg="#1e1e1e")
    frame_controls.pack(pady=10)

    lbl_info = tk.Label(frame_controls, text="", bg="#1e1e1e", fg="white", font=("Arial", 14, "bold"))
    lbl_info.pack()

    # Frame para imágenes
    frame_img = tk.Frame(tab_visual, bg="#1e1e1e")
    frame_img.pack(expand=True)

    img_label = tk.Label(frame_img, bg="#1e1e1e")
    img_label.pack(pady=10)

    # Dropdown con opciones de resultado
    combo_imgs = ttk.Combobox(frame_controls, state="readonly", width=40)
    combo_imgs.pack(pady=5)

    # Botones navegación
    frame_nav = tk.Frame(tab_visual, bg="#1e1e1e")
    frame_nav.pack(pady=15)

    def mostrar_resultado():
        i = index.get()
        ruta, imagenes = lista_resultados[i]
        lbl_info.config(text=f"Imagen {i+1}/{len(lista_resultados)}: {ruta.split('/')[-1]}")

        keys = list(imagenes.keys())
        combo_imgs["values"] = keys

        sel = combo_imgs.get() or keys[0]
        combo_imgs.set(sel)

        im = imagenes[sel]
        mostrar_en_label(im)

    def mostrar_en_label(im):
        im_pil = Image.fromarray(im)
        im_pil.thumbnail((800, 600))
        imtk = ImageTk.PhotoImage(im_pil)
        img_label.config(image=imtk)
        img_label.image = imtk

    def siguiente():
        if index.get() < len(lista_resultados) - 1:
            index.set(index.get() + 1)
        else:
            index.set(0)
        mostrar_resultado()

    def anterior():
        if index.get() > 0:
            index.set(index.get() - 1)
        else:
            index.set(len(lista_resultados) - 1)
        mostrar_resultado()

    combo_imgs.bind("<<ComboboxSelected>>", lambda e: mostrar_resultado())

    tk.Button(frame_nav, text="<< Anterior", command=anterior, bg="#444", fg="white",
              font=("Arial", 11), width=12).pack(side="left", padx=10)
    tk.Button(frame_nav, text="Siguiente >>", command=siguiente, bg="#444", fg="white",
              font=("Arial", 11), width=12).pack(side="left", padx=10)

    # Inicializar primera imagen
    mostrar_resultado()

    # ------------------------------
    # ⚖️ TAB 2: COMPARACIÓN ENTRE IMÁGENES
    # ------------------------------
    tk.Label(tab_comp, text="🔍 Comparación entre vectores de características",
             bg="#1e1e1e", fg="white", font=("Arial", 14, "bold")).pack(pady=15)

    frame_comp = tk.Frame(tab_comp, bg="#1e1e1e")
    frame_comp.pack(pady=10)

    nombres = df["nombre"].tolist()
    combo1 = ttk.Combobox(frame_comp, values=nombres, state="readonly", width=30)
    combo2 = ttk.Combobox(frame_comp, values=nombres, state="readonly", width=30)
    combo1.grid(row=0, column=0, padx=15)
    combo2.grid(row=0, column=1, padx=15)

    lbl_res = tk.Label(tab_comp, text="", bg="#1e1e1e", fg="#00ff99", font=("Arial", 12, "bold"))
    lbl_res.pack(pady=10)

    frame_plot = tk.Frame(tab_comp, bg="#1e1e1e")
    frame_plot.pack(pady=5)

    def comparar():
        idx1, idx2 = combo1.current(), combo2.current()
        if idx1 == -1 or idx2 == -1 or idx1 == idx2:
            messagebox.showwarning("Atención", "Seleccione dos imágenes distintas para comparar.")
            return

        # === Extraer vectores ===
        v1 = df.iloc[idx1].drop("nombre").astype(float).round(4)
        v2 = df.iloc[idx2].drop("nombre").astype(float).round(4)

        # === Cálculos principales ===
        diff = np.abs(v1 - v2).round(4)
        dist_eucl = np.linalg.norm(v1 - v2)
        sim_porcentaje = 100 - (np.mean(diff / (np.maximum(v1, v2) + 1e-6)) * 100)

        # === Mostrar texto ===
        lbl_res.config(
            text=f"📏 Distancia Euclidiana: {dist_eucl:.4f} | "
                 f"🔹 Similitud aproximada: {sim_porcentaje:.2f}%"
        )

        # === Tabla matricial ===
        df_comp = pd.DataFrame({
            "Descriptor": v1.index,
            combo1.get(): v1.values,
            combo2.get(): v2.values,
            "Diferencia Abs.": diff.values
        })

        # Limpiar plots previos
        for widget in frame_plot.winfo_children():
            widget.destroy()

        # === Mostrar DataFrame resumido como tabla ===
        for widget in frame_plot.winfo_children():
            widget.destroy()

        frame_table = tk.Frame(frame_plot, bg="#1e1e1e")
        frame_table.pack(fill="both", expand=True, padx=20, pady=10)

        # Crear Treeview
        cols = ["Descriptor", combo1.get(), combo2.get(), "Diferencia Abs."]
        tree = ttk.Treeview(frame_table, columns=cols, show="headings", height=20)

        # Configurar encabezados
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=180)

        # Scroll vertical
        scroll_y = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")

        # Insertar filas
        for _, row in df_comp.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(fill="both", expand=True)

    tk.Button(tab_comp, text="Comparar", command=comparar,
              bg="#007acc", fg="white", font=("Arial", 12, "bold"), width=15).pack(pady=10)

    ventana.mainloop()