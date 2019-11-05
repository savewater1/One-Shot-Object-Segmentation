# This code is a edited snippet from https://github.com/Yuliang-Zou/tf_fcn

import tensorflow as tf
import numpy as np
from .util import bilinear_upsample_weights

"""Define a base class, containing some useful layer functions"""
class Network(object):
	def __init__(self, inputs):
		self.inputs = []
		self.layers = {}
		self.outputs = {}

	"""Get outputs given key names"""
	def get_output(self, key):
		if key not in self.outputs:
			raise KeyError
		return self.outputs[key]

	"""Get parameters given key names"""
	def get_param(self, key):
		if key not in self.layers:
			raise KeyError
		return self.layers[key]['weights'], self.layers[key]['biases']

	"""Add conv part of vgg16"""
	def add_conv(self, inputs, num_classes, stage='TRAIN'):
		# Dropout is different for training and testing
		if stage == 'TRAIN':
			keep_prob = 0.5
		elif stage == 'TEST':
			keep_prob = 1
		else:
			raise ValueError

		# Conv1
		with tf.variable_scope('conv1_1') as scope:
			w_conv1_1 = tf.get_variable('weights', [3, 3, 3, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_1 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_1 = tf.nn.conv2d(inputs, w_conv1_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_1
			a_conv1_1 = tf.nn.relu(z_conv1_1)

		with tf.variable_scope('conv1_2') as scope:
			w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_2 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_2
			a_conv1_2 = tf.nn.relu(z_conv1_2)
		
		pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool1')

		# Conv2
		with tf.variable_scope('conv2_1') as scope:
			w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_1 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_1
			a_conv2_1 = tf.nn.relu(z_conv2_1)

		with tf.variable_scope('conv2_2') as scope:
			w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_2 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_2
			a_conv2_2 = tf.nn.relu(z_conv2_2)

		pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool2')

		# Conv3
		with tf.variable_scope('conv3_1') as scope:
			w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_1 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_1
			a_conv3_1 = tf.nn.relu(z_conv3_1)

		with tf.variable_scope('conv3_2') as scope:
			w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_2 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_2
			a_conv3_2 = tf.nn.relu(z_conv3_2)

		with tf.variable_scope('conv3_3') as scope:
			w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_3 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_3
			a_conv3_3 = tf.nn.relu(z_conv3_3)

		pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool3')

		# Conv4
		with tf.variable_scope('conv4_1') as scope:
			w_conv4_1 = tf.get_variable('weights', [3, 3, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_1
			a_conv4_1 = tf.nn.relu(z_conv4_1)

		with tf.variable_scope('conv4_2') as scope:
			w_conv4_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_2 = tf.nn.conv2d(a_conv4_1, w_conv4_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_2
			a_conv4_2 = tf.nn.relu(z_conv4_2)

		with tf.variable_scope('conv4_3') as scope:
			w_conv4_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_3 = tf.nn.conv2d(a_conv4_2, w_conv4_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_3
			a_conv4_3 = tf.nn.relu(z_conv4_3)

		pool4 = tf.nn.max_pool(a_conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool4')

		# Conv5
		with tf.variable_scope('conv5_1') as scope:
			w_conv5_1 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_1
			a_conv5_1 = tf.nn.relu(z_conv5_1)

		with tf.variable_scope('conv5_2') as scope:
			w_conv5_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_2 = tf.nn.conv2d(a_conv5_1, w_conv5_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_2
			a_conv5_2 = tf.nn.relu(z_conv5_2)

		with tf.variable_scope('conv5_3') as scope:
			w_conv5_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_3 = tf.nn.conv2d(a_conv5_2, w_conv5_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_3
			a_conv5_3 = tf.nn.relu(z_conv5_3)

		pool5 = tf.nn.max_pool(a_conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool5')

		# Transform fully-connected layers to convolutional layers
		with tf.variable_scope('conv6') as scope:
			w_conv6 = tf.get_variable('weights', [7, 7, 512, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv6 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv6 = tf.nn.conv2d(pool5, w_conv6, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv6
			a_conv6 = tf.nn.relu(z_conv6)
			d_conv6 = tf.nn.dropout(a_conv6, keep_prob)

		with tf.variable_scope('conv7') as scope:
			w_conv7 = tf.get_variable('weights', [1, 1, 4096, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv7 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv7 = tf.nn.conv2d(d_conv6, w_conv7, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv7
			a_conv7 = tf.nn.relu(z_conv7)
			d_conv7 = tf.nn.dropout(a_conv7, keep_prob)

		# Replace the original classifier layer
		with tf.variable_scope('conv8') as scope:
			w_conv8 = tf.get_variable('weights', [1, 1, 4096, num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv8 = tf.get_variable('biases', [num_classes],
				initializer=tf.constant_initializer(0))
			z_conv8 = tf.nn.conv2d(d_conv7, w_conv8, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv8

		# Add to store dicts
		self.outputs['conv1_1'] = a_conv1_1
		self.outputs['conv1_2'] = a_conv1_2
		self.outputs['pool1']   = pool1
		self.outputs['conv2_1'] = a_conv2_1
		self.outputs['conv2_2'] = a_conv2_2
		self.outputs['pool2']   = pool2
		self.outputs['conv3_1'] = a_conv3_1
		self.outputs['conv3_2'] = a_conv3_2
		self.outputs['conv3_3'] = a_conv3_3
		self.outputs['pool3']   = pool3
		self.outputs['conv4_1'] = a_conv4_1
		self.outputs['conv4_2'] = a_conv4_2
		self.outputs['conv4_3'] = a_conv4_3
		self.outputs['pool4']   = pool4
		self.outputs['conv5_1'] = a_conv5_1
		self.outputs['conv5_2'] = a_conv5_2
		self.outputs['conv5_3'] = a_conv5_3
		self.outputs['pool5']   = pool5
		self.outputs['conv6']   = d_conv6
		self.outputs['conv7']   = d_conv7
		self.outputs['conv8']   = z_conv8

		self.layers['conv1_1'] = {'weights':w_conv1_1, 'biases':b_conv1_1}
		self.layers['conv1_2'] = {'weights':w_conv1_2, 'biases':b_conv1_2}
		self.layers['conv2_1'] = {'weights':w_conv2_1, 'biases':b_conv2_1}
		self.layers['conv2_2'] = {'weights':w_conv2_2, 'biases':b_conv2_2}
		self.layers['conv3_1'] = {'weights':w_conv3_1, 'biases':b_conv3_1}
		self.layers['conv3_2'] = {'weights':w_conv3_2, 'biases':b_conv3_2}
		self.layers['conv3_3'] = {'weights':w_conv3_3, 'biases':b_conv3_3}
		self.layers['conv4_1'] = {'weights':w_conv4_1, 'biases':b_conv4_1}
		self.layers['conv4_2'] = {'weights':w_conv4_2, 'biases':b_conv4_2}
		self.layers['conv4_3'] = {'weights':w_conv4_3, 'biases':b_conv4_3}
		self.layers['conv5_1'] = {'weights':w_conv5_1, 'biases':b_conv5_1}
		self.layers['conv5_2'] = {'weights':w_conv5_2, 'biases':b_conv5_2}
		self.layers['conv5_3'] = {'weights':w_conv5_3, 'biases':b_conv5_3}
		self.layers['conv6']   = {'weights':w_conv6, 'biases':b_conv6}
		self.layers['conv7']   = {'weights':w_conv7, 'biases':b_conv7}
		self.layers['conv8']   = {'weights':w_conv8, 'biases':b_conv8}


"""Baseline model"""
class FCN32(Network):
	def __init__(self, inp_size, no_of_classes, inp_img=None, mode="TEST"):
		self.num_classes = no_of_classes
		self.max_size = inp_size
		
		if inp_img is not None:
			assert inp_img.shape[1:] == [self.max_size[0], self.max_size[1], 3]
			self.img = inp_img
		else:
			self.img  = tf.placeholder(tf.float32, [None, self.max_size[0], self.max_size[1], 3])

		self.layers = {}
		self.outputs = {}
		self.set_up(mode)

	def set_up(self, mode="TEST"):
		self.add_conv(self.img, self.num_classes, stage=mode)
		self.add_deconv(bilinear=False)

	"""Add the deconv(upsampling) layer to get dense prediction"""
	def add_deconv(self, bilinear=False):
		conv8 = self.get_output('conv8')

		with tf.variable_scope('deconv') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [64, 64, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(32, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(conv8, w_deconv, 
				[tf.shape(self.img)[0], self.max_size[0], self.max_size[1], self.num_classes],
				strides=[1,32,32,1], padding='SAME', name='z') + b_deconv

		# Add to store dicts
		self.outputs['deconv'] = z_deconv
		self.layers['deconv']  = {'weights':w_deconv, 'biases':b_deconv}

