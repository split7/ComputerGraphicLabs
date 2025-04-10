import math

import numpy as np
from PIL import Image, ImageOps

from lab2.main import normal, barycentric
from labs1.draw import parseV, parseF

def draw_trig(img, color, zz_buf, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = 25000
    b = 2000
    x0_proj = a * x0 / z0 + b
    y0_proj = a * y0 / z0 + b
    x1_proj = a * x1 / z1 + b
    y1_proj = a * y1 / z1 + b
    x2_proj = a * x2 / z2 + b
    y2_proj = a * y2 / z2 + b


    xmin = math.floor(min(x0_proj, x1_proj, x2_proj))
    xmax = math.ceil(max(x0_proj, x1_proj, x2_proj))
    ymin = math.floor(min(y0_proj, y1_proj, y2_proj))
    ymax = math.ceil(max(y0_proj, y1_proj, y2_proj))

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0



    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l1, l2, l3 = barycentric(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l1 > 0 and l2 > 0 and l3 > 0:
                z = l1 * z0 + l2 * z1 + l3 * z2
                if z < zz_buf[y, x]:
                    zz_buf[y, x] = z
                    img[y, x] = color

def rotate(coordinates: np.array, alpha: float, beta: float, gamma: float) -> np.array:
    x_rotate = np.array([[1, 0, 0],
                         [0, np.cos(alpha), np.sin(alpha)],
                         [0, -np.sin(alpha), np.cos(alpha)]])
    y_rotate = np.array([[np.cos(beta), 0, np.sin(beta)],
                         [0, 1, 0],
                         [-np.sin(beta), 0, np.cos(beta)]])
    z_rotate = np.array([[np.cos(gamma), np.sin(gamma), 0],
                         [-np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])
    r = x_rotate @ y_rotate @ z_rotate
    new_coordinates = r @ coordinates + np.array([0.01, 0.001, 20])
    return new_coordinates


if __name__ == '__main__':
    v_obj = parseV('model_1.obj')
    f_obj = parseF('model_1.obj')
    img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
    z_buf = np.full((4000, 4000), np.inf, dtype=np.float32)

    for cord in v_obj:
        cord[0], cord[1], cord[2] = rotate(np.array([cord[0], cord[1], cord[2]]), 60, 60, 0)

    for i in f_obj:
        v1 = v_obj[i[0] - 1]
        v2 = v_obj[i[1] - 1]
        v3 = v_obj[i[2] - 1]
        l = np.array([0, 0, 1])
        n = normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
        scalar = np.dot(n, l) / np.linalg.norm(n)
        if scalar < 0:
            draw_trig(img_mat, [-255 * scalar, -120 * scalar, -200 * scalar], z_buf,
                      v1[0], v1[1], v1[2],
                      v2[0], v2[1], v2[2],
                      v3[0], v3[1], v3[2])

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('image_rotate10.png')