import math

import numpy as np
from PIL import Image, ImageOps

from lab2.main import normal, barycentric
from labs1.draw import parseV
from labs3.main import rotate

def draw_trig(img, l, zz_buf, x0, y0, z0, x1, y1, z1, x2, y2, z2, vn_calc, texture, images):
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

    i0 = np.dot(vn_calc[0], l) / (np.linalg.norm(vn_calc[0]) * np.linalg.norm(l))
    i1 = np.dot(vn_calc[1], l) / (np.linalg.norm(vn_calc[1]) * np.linalg.norm(l))
    i2 = np.dot(vn_calc[2], l) / (np.linalg.norm(vn_calc[2]) * np.linalg.norm(l))

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l1, l2, l3 = barycentric(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, x, y)
            if l1 > 0 and l2 > 0 and l3 > 0:
                z = l1 * z0 + l2 * z1 + l3 * z2
                if z < zz_buf[y, x]:
                    zz_buf[y, x] = z
                    u = round(1024 * (l1 * texture[0][0] + l2 * texture[1][0] + l3 * texture[2][0]))
                    v = round(1024 * (l1 * texture[0][1] + l2 * texture[1][1] + l3 * texture[2][1]))
                    color = images.getpixel((u, v))
                    k = - (i0*l1 + i1*l2 + i2*l3)
                    img[y,x] = (color[0] * k, color[1] * k, color[2] * k)

def parseF(name: str):
    with open(name) as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if line[0] == 'f' and line[1] == ' ':
                listV = line[2:len(line) - 1].split(' ')
                v = []
                for l in listV:
                    p = l.split('/')
                    v.append([int(p[0]), int(p[1])])
                list.append(v)
        return list

def parseVT(name: str):
    with open(name) as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if line[0] == 'v' and line[1] == 't':
                list.append([float(x) for x in line[3:len(line) - 1].split(' ')])
        return list


if __name__ == '__main__':
    images = Image.open('bunny-atlas.jpg')
    images = ImageOps.flip(images)
    v_obj = parseV('model_1.obj')
    f_obj = parseF('model_1.obj')
    vt_obj = parseVT('model_1.obj')
    img_mat = np.zeros((5000, 5000, 3), dtype=np.uint8)
    z_buf = np.full((5000, 5000), np.inf, dtype=np.float32)
    vn_calc = np.zeros((len(v_obj), 3), dtype=np.float32)
    for cord in v_obj:
        cord[0], cord[1], cord[2] = rotate(np.array([cord[0], cord[1], cord[2]]), 0, 90, 0)

    for i in f_obj:
        v1 = v_obj[i[0][0] - 1]
        v2 = v_obj[i[1][0] - 1]
        v3 = v_obj[i[2][0] - 1]
        n0 = normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])

        vn_calc[i[0][0] - 1] += n0 / np.linalg.norm(n0)
        vn_calc[i[1][0] - 1] += n0 / np.linalg.norm(n0)
        vn_calc[i[2][0] - 1] += n0 / np.linalg.norm(n0)

    for i in range(len(vn_calc)):
        vn_calc[i] = vn_calc[i] / np.linalg.norm(vn_calc[i])

    for i in f_obj:
        v1 = v_obj[i[0][0] - 1]
        v2 = v_obj[i[1][0] - 1]
        v3 = v_obj[i[2][0] - 1]
        l = np.array([0, 0, 1])
        n = normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
        scalar = np.dot(n, l) / np.linalg.norm(n)
        if scalar < 0:
            draw_trig(img_mat, l, z_buf,
                      v1[0], v1[1], v1[2],
                      v2[0], v2[1], v2[2],
                      v3[0], v3[1], v3[2],np.array([vn_calc[i[0][0] - 1], vn_calc[i[1][0] - 1], vn_calc[i[2][0] - 1]]), np.array([vt_obj[i[0][1] - 1], vt_obj[i[1][1] - 1], vt_obj[i[2][1] - 1]]),images)

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('image4.png')
