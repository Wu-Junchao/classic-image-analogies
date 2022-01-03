import numpy as np
import cv2
from numpy.linalg import norm

def cvt2yuv(img):
	return cv2.cvtColor(img.astype('float32'),cv2.COLOR_RGB2YUV)

def resize_img(img,scale_percent):

    height = int(img.shape[0] * scale_percent)
    width = int(img.shape[1] * scale_percent)
    return cv2.resize(img,(width, height),interpolation = cv2.INTER_AREA)

def get_pyramid(img,level):
    image_pyramid = [img.copy()]

    while (level>0):
        image_pyramid.append(cv2.pyrDown(image_pyramid[-1]))
        level=level-1

    image_pyramid.reverse()
    return image_pyramid

def concat_features(pyramid):
    layer = len(pyramid)
    features = [pyramid[i][:,:,0].copy() for i in range(layer)]

    F = [[]]
    SMALL_N = 3
    BIG_N = 5
    small_pad = SMALL_N//2
    big_pad = BIG_N//2

    for i in range(1,layer):
        fl = np.zeros(features[i].shape+(SMALL_N**2+BIG_N**2,))
        small_padded = np.pad(features[i-1], (small_pad,), 'reflect')
        big_padded = np.pad(features[i], (big_pad,), 'reflect')

        for x in range(features[i].shape[0]):
            for y in range(features[i].shape[1]):
                fl[x,y,:] = get_features(small_padded,big_padded,x,y)
        F.append(np.vstack(fl))

    return F


def get_features(small_padded, big_padded, row, col):
    SMALL_N = 3
    BIG_N = 5
    
    small_patch = small_padded[row//2 : row//2 + SMALL_N, col//2 : col//2 + SMALL_N]
    big_patch = big_padded[row : row + BIG_N, col : col + BIG_N]
    features = np.hstack([small_patch.flatten(), big_patch.flatten()])

    return features

def remap_y(a, ap, b):
    ya = a[:,:,0]
    yap = ap[:,:,0]
    yb = b[:,:,0]
    
    mean_a = np.mean(ya)
    std_a = np.std(ya)
    mean_b = np.mean(yb)
    std_b = np.std(yb)

    a[:,:,0] = (std_b/std_a) * (ya - mean_a) + mean_b
    ap[:,:,0] = (std_b/std_a) * (yap - mean_a) + mean_b
    return a, ap