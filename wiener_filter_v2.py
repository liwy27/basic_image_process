from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from numba import jit
import time
from functools import wraps
import cv2 as cv
import numpy as np
# from numpy.fft import fft2, ifft2, fftshift

obj_path = './data/20201128/mid_ring/mid_ring_220ms_pos.bmp'
psf_path = './data/20201128/point/point_550ms_pos.bmp'
USE_CUDA = False

if USE_CUDA:
    import cupy as cp
    from cupy.fft import fft2, ifft2, fftshift
    from cupy import abs, conj
else:
    from numpy import abs, conj
    from scipy.fftpack import fft2, ifft2, fftshift


def time_cost(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        a = time.time()
        func_out = func(*args, **kwargs)
        print("total time:{:.4f}s".format(time.time()-a))
        return func_out
    return wrapped


def inverse(input, PSF, eps):
    input_fft = fft2(input)
    PSF_fft = fft2(PSF) + eps #eps:噪声功率
    result = ifft2(input_fft / PSF_fft)
    result = np.abs(fftshift(result))
    return result


@time_cost
def wiener(inp, psf, eps=0, k=10000000.):
    input_fft = fft2(inp)
    psf_fft = fft2(psf) + eps
    psf_fft = conj(psf_fft) / (abs(psf_fft)**2 + k)
    result = ifft2(input_fft * psf_fft)
    result = abs(fftshift(result))
    # result = np.abs(result)
    return result


@jit(nopython=True)
def clip(array):
    array = np.where(array < 0,  0, array)
    array = np.where(array > 255, 255, array)
    return array.astype(np.uint8)


@jit(nopython=True)
def normalize(array):
    max_value = np.max(array)
    min_value = np.min(array)
    return (array - min_value) / (max_value - min_value)


diffused_src = cv.imread(obj_path)
# diffused_tem = cv.split(diffused_src)
# diffused = diffused_tem[0]

diffused = cv.cvtColor(diffused_src, cv.COLOR_BGR2GRAY)
diffused = normalize(diffused)
plt.figure(figsize=(16, 16))
plt.xlabel('diffused')
plt.subplot(221)
im = plt.imshow(diffused, cmap=cm.hot)
# plt.colorbar(im)


PSF_src = cv.imread(psf_path)
# PSF_tem = cv.split(PSF_src)
# PSF = PSF_tem[0]

PSF = cv.cvtColor(PSF_src, cv.COLOR_BGR2GRAY)
PSF = normalize(PSF)
plt.xlabel('PSF')
plt.subplot(222)
im = plt.imshow(PSF, cmap=cm.hot)
# plt.colorbar(im)

if USE_CUDA:
    diffused = cp.array(diffused)
    PSF = cp.array(PSF)

reconstruct = wiener(diffused, PSF, 0)
reconstruct = normalize(reconstruct)

out = 255*reconstruct
out = clip(out)
cv.imwrite('out.jpg', out)

plt.xlabel('reslut')
plt.subplot(212)
im = plt.imshow(reconstruct, cmap=cm.hot)
plt.colorbar(im)
plt.show()
print("ok!you are very good!")
