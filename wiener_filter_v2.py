from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import math
import cv2 as cv
from scipy.fftpack import fft2, ifft2, fftshift


def inverse(input, PSF, eps):
    input_fft = fft2(input)
    PSF_fft = fft2(PSF) + eps #eps:噪声功率
    result = ifft2(input_fft / PSF_fft)
    result = np.abs(fftshift(result))
    return result


def wiener(input, PSF, eps, K=100.):
    input_fft = fft2(input)
    PSF_fft = fft2(PSF)
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + K)
    result = ifft2(input_fft * PSF_fft_1)
    result = np.abs(fftshift(result))
    # result = np.abs(result)
    return result


def normal(array):
    array = np.where(array < 0,  0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.uint8)
    return array


diffused_src = cv.imread('./wu_exam_2.png')
# diffused_tem = cv.split(diffused_src)
# diffused = diffused_tem[0]

diffused = cv.cvtColor(diffused_src, cv.COLOR_BGR2GRAY)
# diffused = diffused / 255.
max_value = np.max(diffused)
min_value = np.min(diffused)
diffused = (diffused - min_value) / (max_value - min_value)
plt.figure(1)
plt.xlabel('diffused')
im = plt.imshow(diffused, cmap=cm.hot)
plt.colorbar(im)
plt.show()

PSF_src = cv.imread('./point_exam_2.png')
# PSF_tem = cv.split(PSF_src)
# PSF = PSF_tem[0]
PSF = cv.cvtColor(PSF_src, cv.COLOR_BGR2GRAY)
# PSF = PSF / 255.
max_value = np.max(PSF)
min_value = np.min(PSF)
PSF = (PSF - min_value) / (max_value - min_value)

plt.figure(2)
plt.xlabel('PSF')
im = plt.imshow(PSF, cmap=cm.hot)
plt.colorbar(im)
plt.show()

reconstruct = wiener(diffused, PSF, 0)
max_value = np.max(reconstruct)
min_value = np.min(reconstruct)
reconstruct = (reconstruct - min_value) / (max_value - min_value)
# reconstruct = reconstruct/max_value

out = 255*reconstruct
out = normal(out)
cv.imwrite('out.jpg', out)

cv.imshow("out", out)
cv.waitKey(10000)
plt.figure(3)
plt.xlabel('reslut')
im = plt.imshow(reconstruct, cmap=cm.hot)
plt.colorbar(im)
plt.show()


