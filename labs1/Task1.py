import numpy as np
from PIL import Image

img_mat = np.zeros((600, 200, 3), dtype=np.uint8)
img_mat[0:600, 0:800, 2] = 255
for i in range(600):
    for j in range(200):
        img_mat[i, j, 0] = i / 4
        img_mat[i, j, 1] = j / 2
        img_mat[i, j, 2] = (i + j) / 4


img = Image.fromarray(img_mat, mode='RGB') # 'L' - полутон, 'RGB' - цвет
img.save('image2.png')