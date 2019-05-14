# this code for load fast mc-cnn model

import tensorflow as tf
import numpy as np
import cv2


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

#here, None of x is the number of total img of a img
x = tf.placeholder("float32", shape=[None, 11, 11])

x = tf.reshape(x, [-1, 11, 11, 1])

#the first layer
w_conv1=weight_variable([3, 3, 1, 64])
b_conv1=bias_variable([64])
h_conv1=tf.nn.relu(conv2d(x, w_conv1)+b_conv1)

#the second layer
w_conv2=weight_variable([3,3,64,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_conv1, w_conv2)+b_conv2)

#the third layer
w_conv3=weight_variable([3,3,64,64])
b_conv3=bias_variable([64])
h_conv3=tf.nn.relu(conv2d(h_conv2, w_conv3)+b_conv3)

#the fourth layer
w_conv4=weight_variable([3,3,64,64])
b_conv4=bias_variable([64])
h_conv4=tf.nn.relu(conv2d(h_conv3, w_conv4)+b_conv4)

#the fifth layer, without relu
w_conv5=weight_variable([3,3,64,64])
b_conv5=bias_variable([64])
h_conv5=conv2d(h_conv4, w_conv5)+b_conv5


#normalize
h_conv5 = tf.reshape(h_conv5, [-1, 64])
vec = tf.nn.l2_normalize(h_conv5, axis=1)  #dim -> axis

#================================================================

class Model(object):

	def __init__(self, model_path):
		self._model_path = model_path
		self._pad_value = 0

	def __load_model(self):
		# tf.InteractiveSession().close()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		#load model
		saver = tf.train.Saver()
		model = tf.train.get_checkpoint_state(self._model_path)
		if model and model.model_checkpoint_path:
			saver.restore(sess, model.model_checkpoint_path)
			return True, sess
		else:
			print("load model error")
			return False, sess

	def __get_net_result(self, img_pad, row_num):

		flag, sess = self.__load_model()
		if not flag:
			return

		height = img_pad.shape[0] -10
		width = img_pad.shape[1]-10

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
		feature_vector = vec.eval(feed_dict = {x: net_input}, session=sess)
		#print("here=================================================")
		return feature_vector

	def get_feature_vector(self, img_pad, row_num):
		feature_vector = self.__get_net_result(img_pad, row_num)
		return feature_vector