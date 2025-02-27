import numpy as np
from PIL import Image, ImageOps
import math

from labs1.draw import parseV, parseF


def barycentric(x0, y0, x1, y1, x2, y2, x, y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_trig(img, color, zz_buf, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    xmin = math.floor(min(x0, x1, x2))
    xmax = math.ceil(max(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    ymax = math.ceil(max(y0, y1, y2))

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l1, l2, l3 = barycentric(x0, y0, x1, y1, x2, y2, x, y)
            if l1 > 0 and l2 > 0 and l3 > 0:
                z = l1 * z0 + l2 * z1 + l3 * z2
                if z < zz_buf[y, x]:
                    zz_buf[y, x] = z
                    img[y, x] = color


def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))


if __name__ == '__main__':
    v_obj = parseV('model_1.obj')
    f_obj = parseF('model_1.obj')
    img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
    z_buf = np.full((4000, 4000), np.inf, dtype=np.float32)
    for i in f_obj:
        v1 = v_obj[i[0] - 1]
        v2 = v_obj[i[1] - 1]
        v3 = v_obj[i[2] - 1]
        l = np.array([0, 0, 1])
        n = normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
        scalar = np.dot(n, l) / np.linalg.norm(n)
        if scalar < 0:
            draw_trig(img_mat, [-255 * scalar, 120, 200], z_buf,
                      (v1[0] * 20000 + 2156.66), (v1[1] * 20000 + 1204.04), (v1[2] * 20000),
                      (v2[0] * 20000 + 2156.66), (v2[1] * 20000 + 1204.04), (v2[2] * 20000),
                      (v3[0] * 20000 + 2156.66), (v3[1] * 20000 + 1204.04), (v3[2] * 20000))

    img = Image.fromarray(img_mat, mode='RGB')  # 'L' - полутон, 'RGB' - цвет
    img = ImageOps.flip(img)
    img.save('image_triangle5.png')