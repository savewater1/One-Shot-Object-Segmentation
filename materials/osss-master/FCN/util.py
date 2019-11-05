# This code is a snippet from https://github.com/Yuliang-Zou/tf_fcn

import numpy as np
import cv2

"""
Helper functions for bilinear upsampling
credit: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
"""
def get_kernel_size(factor):
	"""
	Find the kernel size given the desired factor of upsampling.
	"""
	return 2 * factor - factor % 2

def upsample_filt(size):
	"""
	Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
	"""
	factor = (size + 1) // 2
	if size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:size, :size]
	return (1 - abs(og[0] - center) / factor) * \
			(1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
	"""
	Create weights matrix for transposed convolution with bilinear filter
	initialization.
	"""
	filter_size = get_kernel_size(factor)

	weights = np.zeros((filter_size,
						filter_size,
						number_of_classes,
						number_of_classes), dtype=np.float32)

	upsample_kernel = upsample_filt(filter_size)

	for i in xrange(number_of_classes): 
		weights[:, :, i, i] = upsample_kernel

	return weights