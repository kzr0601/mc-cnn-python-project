# this code for load fast mc-cnn model

import tensorflow as tf
import numpy as np
import cv2
import random


#============== net architecture copy from net.py ==================
#   get the normalized feature vector of all img of an img

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

#here, None is batchSize
x_L = tf.placeholder("float32", shape=[None, 11, 11])

x_L = tf.reshape(x_L, [-1, 11, 11, 1])

#the first layer
with tf.name_scope('conv1'):
	w_conv1=weight_variable([3, 3, 1, 112])
	b_conv1=bias_variable([112])
	h_conv1=tf.nn.relu(conv2d(x_L, w_conv1)+b_conv1)

#the second layer
with tf.name_scope('conv2'):
	w_conv2=weight_variable([3,3,112,112])
	b_conv2=bias_variable([112])
	h_conv2=tf.nn.relu(conv2d(h_conv1, w_conv2)+b_conv2)

#the third layer
with tf.name_scope('conv3'):
	w_conv3=weight_variable([3,3,112,112])
	b_conv3=bias_variable([112])
	h_conv3=tf.nn.relu(conv2d(h_conv2, w_conv3)+b_conv3)

#the fourth layer
with tf.name_scope('conv4'):
	w_conv4=weight_variable([3,3,112,112])
	b_conv4=bias_variable([112])
	h_conv4=tf.nn.relu(conv2d(h_conv3, w_conv4)+b_conv4)

#the fifth layer, without relu
with tf.name_scope('conv5'):
	w_conv5=weight_variable([3,3,112,112])
	b_conv5=bias_variable([112])
	h_conv5=tf.nn.relu(conv2d(h_conv4, w_conv5)+b_conv5)
	#concatenate
	h_conv5 = tf.reshape(h_conv5, [-1, 112])




class Model(object):

	def __init__(self, model_path):
		self._model_path = model_path
		self._pad_value = 0

	def __load_model(self):

		# tf.InteractiveSession().close()
		#sess = tf.InteractiveSession()
		sess = tf.Session() #config=tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)))
		sess.run(tf.global_variables_initializer())

		#load model
		variables = tf.contrib.framework.get_variables_to_restore()
		variables_to_restore = [v for v in variables if v.name.split('/')[0][0:4]=='conv']
		saver = tf.train.Saver(variables_to_restore)
		# saver.restore(sess,'model.ckpt')

		# saver = tf.train.Saver()
		model = tf.train.get_checkpoint_state(self._model_path)
		if model and model.model_checkpoint_path:
			saver.restore(sess, model.model_checkpoint_path)
			return True, sess
		else:
			print("load model error")
			return False, sess

	def __get_net_result(self, img_path, row_num):

		flag, sess = self.__load_model()
		if not flag:
			return

		#preprocess img
		img = cv2.imread(img_path, 0).astype("float32")
		height = img.shape[0]
		width = img.shape[1]
		mean = np.mean(img)
		stddev = np.std(img)
		# img = img.astype(np.float32)
		img = (img - mean) / stddev

		# img_pad = np.pad(img, 5, 'constant', constant_values = self._pad_value)
		# edge 'linear_ramp'
		img_pad = np.pad(img, 5, 'edge')

		# fectch patches from img, [none, 11, 11, 1] -> [none, 11, 11]
		net_input = np.empty(( width, 11, 11, 1), dtype=np.float32)
		count = 0
		for c in range(0, width):
			p = img_pad[row_num: row_num+11]
			patch = p[:, range(c, c+11)]
			net_input[count, :, :] = np.reshape(patch, (11, 11, 1))
			count += 1
		#print((height-10)*(width-10) == count)

		#load data into net to get feature vector
		feature_vector = h_conv5.eval(feed_dict = {x_L: net_input}, session=sess)
		# print("here=================================================")
		# print(feature_vector.shape[0])
		return feature_vector


	def get_feature_vector(self, img_path, row_num):
		feature_vector = self.__get_net_result(img_path, row_num)
		return feature_vector

	def get_vector(self, patch):
		flag, sess = self.__load_model()
		if not flag:
			return

		patch1 = np.reshape(patch, (1, 11, 11, 1))
		feature_vector = h_conv5.eval(feed_dict = {x_L: patch1}, session=sess)
		return feature_vector


