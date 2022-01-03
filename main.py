import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import norm
from utils import *
from match import *
import imageio
import copy

def start(a1,b1,a2,k,flg):
	L = 2 
	SMALL_N = 3   
	BIG_N = 5  

	a1=cvt2yuv(a1)
	b1=cvt2yuv(b1)
	a2=cvt2yuv(a2)

	if flg:
		a1, a2 = remap_y(a1, a2, b1)

	a_pyramid = get_pyramid(a1, L)
	a_prime_pyramid = get_pyramid(a2, L)
	b_pyramid = get_pyramid(b1, L)
	b_prime_pyramid = copy.deepcopy(b_pyramid)
	
	b_features = concat_features(b_pyramid)
	a_feature = concat_features(a_pyramid)
	ap_feature = concat_features(a_prime_pyramid)

	flann, flann_params, f_p= ann(a_feature, ap_feature)

	result = []

	for level in range(1, len(a_pyramid)):
		s = []

		b_height, b_width = b_pyramid[level].shape[:2]
		a_height, a_width = a_pyramid[level].shape[:2]

		for row in range(b_height):
			for col in range(b_width):

				print('Working on pixel %d/%d, %d/%d, a decade time remaining:(' % (row,b_height,col,b_width))

				luminance_b_p = [b_prime_pyramid[i][:,:,0] for i in range(len(b_prime_pyramid))]
				fine_pad = np.pad(luminance_b_p[level], (BIG_N//2), 'reflect')
				corase_pad = np.pad(luminance_b_p[level-1], (SMALL_N//2), 'reflect') 
				f_q = np.hstack([b_features[level][row*b_width+col, :],
									  get_features(corase_pad, fine_pad, row,col)])

				best_find = best_approximate_match(flann[level], flann_params[level], f_q,a_width)

				if s:
					best_coh = best_coherence_match(f_p[level] , f_q, s, a_height, a_width, row,col, b_width)

					if best_coh[0] != -1:
						f_app = f_p[level][best_find]
						f_coh = f_p[level][best_coh]
						d_app = norm(f_app - f_q)**2
						d_coh = norm(f_coh - f_q)**2

						if d_coh < d_app * (1 + (2**(level-L)*k)):
							best_find = best_coh

				b_prime_pyramid[level][row, col] = a_prime_pyramid[level][best_find[0],best_find[1]]
				s.append(best_find)

		yuv_res = np.dstack([b_prime_pyramid[level][:, :, 0], b_pyramid[level][:, :, 1:]])
		rgb_res = cv2.cvtColor(yuv_res.astype('float32'),cv2.COLOR_YUV2RGB)
		
		rgb_res = np.clip(rgb_res, 0, 1)

		imageio.imsave('output/level_%d_temp.jpg' % level, rgb_res)
		result.append(rgb_res)

	return result

def start2(a1,b1,a2,k,flg):
	L = 2 
	SMALL_N = 3   
	BIG_N = 5  

	a1=cvt2yuv(a1)
	b1=cvt2yuv(b1)
	a2=cvt2yuv(a2)

	if flg:
		a1, a2 = remap_y(a1, a2, b1)

	a_pyramid = get_pyramid(a1, L)
	a_prime_pyramid = get_pyramid(a2, L)
	b_pyramid = get_pyramid(b1, L)
	b_prime_pyramid = copy.deepcopy(b_pyramid)
	
	b_features = concat_features(b_pyramid)
	a_feature = concat_features(a_pyramid)
	ap_feature = concat_features(a_prime_pyramid)

	flann, flann_params, f_p= ann(a_feature, ap_feature)

	result = []

	for level in range(1, len(a_pyramid)):
		s = []

		b_height, b_width = b_pyramid[level].shape[:2]
		a_height, a_width = a_pyramid[level].shape[:2]

		for row in range(b_height):
			for col in range(b_width):

				print('Working on pixel %d/%d, %d/%d, a decade time remaining:(' % (row,b_height,col,b_width))

				luminance_b_p = [b_prime_pyramid[i][:,:,0] for i in range(len(b_prime_pyramid))]
				fine_pad = np.pad(luminance_b_p[level], (BIG_N//2), 'reflect')
				corase_pad = np.pad(luminance_b_p[level-1], (SMALL_N//2), 'reflect') 
				f_q = np.hstack([b_features[level][row*b_width+col, :],
									  get_features(corase_pad, fine_pad, row,col)])

				best_find = best_approximate_match(flann[level], flann_params[level], f_q,a_width)

				if s:
					best_coh = best_coherence_match(f_p[level] , f_q, s, a_height, a_width, row,col, b_width)

					if best_coh[0] != -1:
						f_app = f_p[level][best_find]
						f_coh = f_p[level][best_coh]
						d_app = norm(f_app - f_q)**2
						d_coh = norm(f_coh - f_q)**2

						if d_coh < d_app * (1 + (2**(level-L)*k)):
							best_find = best_coh

				b_prime_pyramid[level][row, col] = a_prime_pyramid[level][best_find[0],best_find[1]]
				s.append(best_find)

		rgb_res = cv2.cvtColor(b_prime_pyramid[level].astype('float32'),cv2.COLOR_YUV2RGB)
		rgb_res = np.clip(rgb_res, 0, 1)
		imageio.imsave('output/level_%d_temp.jpg' % level, rgb_res)
		result.append(rgb_res)

	return result