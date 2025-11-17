#==============================
# PREPROCESAMIENTO DE IMAGENES DE MANZANAS
#==============================

import os
import cv2
import numpy as np

IMGS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'P2')
imgs = []
num_imgs = 3


dir_imgsValidas = [img for img in os.listdir(IMGS_ROOT) if img.endswith(('.png', '.jpg'))]
rnd_IMGs = []
selected_indices = np.random.choice(len(dir_imgsValidas), num_imgs, replace=False)


for idx in selected_indices:
    rnd_IMGs.append(dir_imgsValidas[idx])
    
selected_images = rnd_IMGs

for img_name in selected_images:
    img_path = os.path.join(IMGS_ROOT, img_name)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs.append(gray_img)
