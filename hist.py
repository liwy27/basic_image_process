#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-1 下午10:41
# @Author  : LWY
# @Site    : https://github.com/liwy27/basic_image_process
# @IDE: PyCharm
import cv2 as cv
import numpy as np


def hist(img):
    src = np.copy(img)
    img_flatten = img.flatten()
    img_flatten[0] = 255
    count = np.bincount(img_flatten)
    sum_cum = np.cumsum(count)
    sum_cum = sum_cum / np.max(sum_cum)
    sum_cum *= 255
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            src[i, j] = int(sum_cum[img[i, j]])
    return src


def draw(img, filename, channel=-1):
    img_flatten = img.flatten()
    count = np.bincount(img_flatten)
    count = count / np.max(count)
    count = count * 255
    n = len(count)
    dst = np.full(shape=(256, 256, 3), fill_value=255, dtype=np.uint8)
    color = None
    if channel == 0:
        color = [255, 0, 0]
    elif channel == 1:
        color = [0, 255, 0]
    elif channel == 2:
        color = [0, 0, 255]
    else:
        color = [0, 0, 0]
    for i in range(256):
        if i < n:
            cv.line(dst, (i, 255), (i, 256 - int(count[i])), color)
        else:
            cv.line(dst, (i, 255), (i, 256), 255)
    return dst


img = cv.imread(r"E:\BaiduNetdiskDownload\tid2013\hist.jpg")
img_b, img_g, img_r = cv.split(img)
img_b_2 = hist(img_b)
img_g_2 = hist(img_g)
img_r_2 = hist(img_r)
img_good = cv.merge([img_b_2, img_g_2, img_r_2])
cv.imshow("bad", img)
cv.imshow("good", img_good)

dst_1 = draw(img_g, "img_g_hist", 1)
dst_2 =draw(img_g_2, "img_g_2_hist",1)

cv.imshow("hist_1", dst_1)
cv.imshow("hist_2", dst_2)
cv.waitKey()
cv.imwrite(filename + ".jpg", dst)