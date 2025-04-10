import math

import numpy as np
from PIL import Image, ImageOps

from lab2.main import normal, barycentric
from labs3.main import rotate


def parseV(name: str):
    with open(name) as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if line[0] == 'v' and line[1] == ' ':
                line = line.replace('  ', ' ')
                list.append([float(x) for x in line[2:len(line) - 1].split(' ')])
        return list

def draw_trig(img, l, zz_buf, x0, y0, z0, x1, y1, z1, x2, y2, z2, vn_calc, texture, images):
    a = 10000
    b = 2500
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
    if xmax > img.shape[0]: xmax = img.shape[0]
    if ymax > img.shape[1]: ymax = img.shape[1]

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
                line = line.replace('  ', ' ')
                listV = line[2:len(line) - 1].split(' ')
                q = listV[0].split('/')
                for i in range(1, len(listV) - (1 if len(listV) == 3 else 2)):
                    p = listV[i].split('/')
                    c = listV[i + 1].split('/')
                    if p[1] == '':
                        list.append([int(q[0]), int(p[0]), int(c[0])])
                    else:
                        list.append([[int(q[0]), int(q[1])], [int(p[0]), int(p[1])], [int(c[0]), int(c[1])]])
        return list

def parseVT(name: str):
    with open(name) as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if line[0] == 'v' and line[1] == 't':
                list.append([float(x) for x in line[3:len(line) - 1].split(' ')])
        return list


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def rotate_with_quaternion(coord, axis, theta, dop):
    axis = axis / np.linalg.norm(axis)
    half_theta = theta / 2
    w = np.cos(half_theta)
    x, y, z = axis * np.sin(half_theta)

    q = np.array([w, x, y, z])
    q_conj = np.array([w, -x, -y, -z])

    p = np.array([0.0, coord[0], coord[1], coord[2]])

    rotated = quaternion_multiply(quaternion_multiply(q, p), q_conj)

    return rotated[1:] + dop

def main(filename_obj: str, filename_img:str, img_mat, z_buf, u, theta, dop) :
    images = Image.open(filename_img)
    images = ImageOps.flip(images)
    v_obj = parseV(filename_obj)
    f_obj = parseF(filename_obj)
    vt_obj = parseVT(filename_obj)
    vn_calc = np.zeros((len(v_obj), 3), dtype=np.float32)

    # for cord in v_obj:
    #    cord[0], cord[1], cord[2] = rotate(np.array([cord[0], cord[1], cord[2]]), 0, np.pi / 2, 0)

    for cord in v_obj:
        cord[0], cord[1], cord[2] = rotate_with_quaternion(np.array([cord[0], cord[1], cord[2]]),
                                                           u,
                                                           theta, dop)

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
                      v3[0], v3[1], v3[2], np.array([vn_calc[i[0][0] - 1], vn_calc[i[1][0] - 1], vn_calc[i[2][0] - 1]]),
                      np.array([vt_obj[i[0][1] - 1], vt_obj[i[1][1] - 1], vt_obj[i[2][1] - 1]]), images)


if __name__ == '__main__':
    img_mat = np.zeros((5000, 5000, 3), dtype=np.uint8)
    z_buf = np.full((5000, 5000), np.inf, dtype=np.float32)
    main('12268_banjofrog_v1_L3.obj', '12268_banjofrog_diffuse.jpg', img_mat, z_buf, np.array([0, 1, 0]), np.pi / 2, np.array([0.01, 0.001, 20]))
    main('model_1.obj', 'bunny-atlas.jpg', img_mat, z_buf, np.array([0, 0, 1]), 0, np.array([0.01, 0.001, 1]))
    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save('image7.png')
