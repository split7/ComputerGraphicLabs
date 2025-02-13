import numpy as np
from PIL import Image

img_mat = np.zeros((200, 200), dtype=np.uint8)

for i in range(13):
    x0 = 100
    y0 = 100
    x1 = 100 + 95 * np.cos(i * 2 * np.pi / 13)
    y1 = 100 + 95 * np.sin(i * 2 * np.pi / 13)

img = Image.fromarray(img_mat, mode='L') # 'L' - полутон, 'RGB' - цвет
img.save('imageTask2.png')