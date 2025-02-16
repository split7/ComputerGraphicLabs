import numpy as np
from PIL import Image, ImageOps
import math


def parseV(name: str):
    with open(name) as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if line[0] == 'v' and line[1] == ' ':
                list.append([float(x) for x in line[2:len(line) - 1].split(' ')])
        return list


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
                    v.append(int(p[0]))
                list.append(v)
        return list

    # '15257/16314/15247', '1685/16320/1685', '1675/16323/1675'


def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0.0, 1.0, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def dotted_line_v2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0.0, 1.0, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line(image, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_hotfix_1(image, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_hotfix_2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range(round(x0), round(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(round(x0), round(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v2_no_y_calc(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > 0.5:
            derror -= 1.0
            y += y_update


def x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > 2.0 * (x1 - x0) * 0.5:
            derror -= 2.0 * (x1 - x0) * 1.0
            y += y_update


def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update


img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
v_obj = parseV('model_1.obj')
f_obj = parseF('model_1.obj')

for i in f_obj:
    v1 = v_obj[i[0] - 1]
    v2 = v_obj[i[1] - 1]
    v3 = v_obj[i[2] - 1]
    bresenham_line(img_mat, round(v1[0] * 20000 +2000), round(v1[1] * 20000 + 2000), round(v2[0] * 20000 + 2000), round(v2[1] * 20000 + 2000), (255, 0, 0))
    bresenham_line(img_mat, round(v1[0] * 20000 +2000), round(v1[1] * 20000 + 2000), round(v3[0]*20000 + 2000), round(v3[1]*20000 + 2000), (255, 0, 0))
    bresenham_line(img_mat,  round(v2[0] * 20000 + 2000),  round(v2[1] * 20000 + 2000),  round(v3[0]*20000 + 2000), round(v3[1]*20000 + 2000), (255, 0, 0))




#for i in v_obj:
   # img_mat[round(5000 * i[1] + 500),round(5000 * i[0] + 500)] = 255

img = Image.fromarray(img_mat, mode='RGB')  # 'L' - полутон, 'RGB' - цвет
img = ImageOps.flip(img)
img.save('imageTask6.png')
