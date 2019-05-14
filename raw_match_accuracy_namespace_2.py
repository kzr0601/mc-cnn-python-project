#L - disp = R

#in model, padding around original img with 5 on each edge

from model_accuracy_namespace import Model
import numpy as np
import cv2
import gc
from pfm import writePfm
import tensorflow as tf
import struct
import os
from datetime import datetime

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')


#concatenate
h_conv5 = tf.placeholder("float32", shape=[None, 112])
h_conv5_R = tf.placeholder("float32", shape=[None, 112])
concat_vector = tf.concat([h_conv5, h_conv5_R], 1)

#the first fully-connected layer
with tf.name_scope('fc1'):
	w_fc1 = weight_variable([224, 384])
	b_fc1 = bias_variable([384])
	h_fc1 = tf.nn.relu(tf.matmul(concat_vector, w_fc1)+b_fc1)
# #dropout
# keep_prob_1 = tf.placeholder(tf.float32)
# h_fc1 = tf.nn.dropout(h_fc1, keep_prob_1)

#the second fully-connected layer
with tf.name_scope('fc2'):
	w_fc2 = weight_variable([384, 384])
	b_fc2 = bias_variable([384])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
# #dropout
# keep_prob_2 = tf.placeholder(tf.float32)
# h_fc2 = tf.nn.dropout(h_fc2, keep_prob_2)

#the third fully-connected layer
with tf.name_scope('fc3'):
	w_fc3 = weight_variable([384, 1])
	b_fc3 = bias_variable([1])
	h_fc3 = tf.matmul(h_fc2, w_fc3)+b_fc3
	
output_score = tf.sigmoid(h_fc3)

class FC_model(object):

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
		variables_to_restore = [v for v in variables if v.name.split('/')[0][0:2]=='fc']
		saver = tf.train.Saver(variables_to_restore)
		model = tf.train.get_checkpoint_state(self._model_path)
		if model and model.model_checkpoint_path:
			saver.restore(sess, model.model_checkpoint_path)
			return True, sess
		else:
			print("load model error")
			return False, sess

	def __get_net_result(self, feature_vector_L, feature_vector_R):

		flag, sess = self.__load_model()
		if not flag:
			return

		result_vector = output_score.eval(feed_dict = {h_conv5: feature_vector_L, h_conv5_R: feature_vector_R}, session=sess)
		#print("here=================================================")
		return result_vector


	def get_result(self, feature_vector_L, feature_vector_R):
		# result_vector [None, 1]
		result_vector = self.__get_net_result(feature_vector_L, feature_vector_R)
		return result_vector

def getMatchCostVolume(d, width, feature_mat_L, feature_mat_R, left_or_right, model_path):
	fc_model = FC_model(model_path)

	if left_or_right:
		feature_vector_L = feature_mat_L
		feature_vector_R = np.ones([width, 112])
		for c in range(d, width):
			feature_vector_R[c, :] = feature_mat_R[c-d]

		mcv_d = fc_model.get_result(feature_vector_L, feature_vector_R)
		for c in range(0, d):
			mcv_d[c] = np.min(mcv_d)-1

	else:
		feature_vector_R = feature_mat_R
		feature_vector_L = np.ones([width, 112])
		for c in range(0, width-d):
			feature_vector_L[c, :] = feature_mat_L[c+d]

		mcv_d = fc_model.get_result(feature_vector_L, feature_vector_R)
		for c in range(width-d, width):
			mcv_d[c] = np.min(mcv_d)-1

	return np.reshape(mcv_d, [width])

	

# left_or_right = True: choose left as referenced img
# left_or_right = False: choose right as referenced img
def main(imgL_path, imgR_path, model_path ,dmax, acrt_path, raw_disp_path, pfm_path, left_or_right, fp):

	#load model
	my_model = Model(model_path)
	#get img size
	img = cv2.imread(imgL_path, 0)
	height = img.shape[0]
	width = img.shape[1]

	#initialize to be -2
	matching_cost_volume = np.ones([dmax, height, width], dtype='float32')
	matching_cost_volume = -2 * matching_cost_volume
	raw_disparity = np.empty([height, width], dtype='float32')

	#compute matching cost upon feature vector using dot mult, using neg matching cost
	for r in range(0, height):
		fp.write(str(r)+": "+str(datetime.now())+"\n")
		fp.flush()
		feature_mat_L = my_model.get_feature_vector(imgL_path, r)
		feature_mat_R = my_model.get_feature_vector(imgR_path, r)
		for d in range(dmax):
			matching_cost_volume[d, r, :] = getMatchCostVolume(d, width, feature_mat_L, feature_mat_R, left_or_right, model_path)

	print(str(r)+" "+str(d))

	#adjust acrt, restrict to [0, 1], and smaller acrt correspondings to better matching
	max_ = np.max(matching_cost_volume)
	min_ = np.min(matching_cost_volume)
	matching_cost_volume = np.array( (matching_cost_volume-min_)/(max_-min_) , dtype='float32')
	matching_cost_volume = -matching_cost_volume

	#write file
	acrt_fp = open(acrt_path, 'wb')
	for d in range(dmax):
		for r in range(height):
			for c in range(width):
				a = struct.pack('f', matching_cost_volume[d][r][c])
				acrt_fp.write(a)

	for r in range(height):
		for c in range(width):
			min_ = 2
			for d in range(dmax):
				if(matching_cost_volume[d][r][c] < min_):
					min_ = matching_cost_volume[d][r][c]
					raw_disparity[r][c] = d

	cv2.imwrite(raw_disp_path, raw_disparity)
	writePfm(raw_disparity, pfm_path)



if __name__ == "__main__" :
	#dmax is the given maximum possible disparity
	dmax = 128 #int(input("please input the maximum disparity: "))
	imgL_path = "./Teddy/im0.png"
	imgR_path = "./Teddy/im1.png"
	model_path = "./checkpoint_accuracy_100w_10w_namespace"

	path = "./result_Teddy_accuracy_namespace"
	acrt_path = path+"/im0.acrt"
	raw_disp_path = path+"/raw_disp.png"
	pfm_path = path+"/raw_disp.pfm"

	if not os.path.exists(path):
		os.makedirs(path)
	
	fp = open(path+"/time.txt", "w")
	
	start = datetime.now()
	fp.write(str("start: ")+str(start)+"\n")
	fp.flush()
	main( imgL_path, imgR_path, model_path, dmax, acrt_path, raw_disp_path, pfm_path, True, fp)
	end = datetime.now()
	fp.write(str("end: ")+str(end)+"\n")
	fp.write(str("use time: ")+str(end-start ))
	fp.close()


	print("end")



# #test
# imgpath = "D:/kzr/mc_cnn_project/MiddEval3-data-H/MiddEval3/trainingH/ArtL/im0.png"
# modelpath = "D:/kzr/CNN/myModel_1221_data2W_ite20W"
# test = Model(modelpath)
# feature1 = test.get_feature_vector(imgpath)

# imgpath = "D:/kzr/mc_cnn_project/MiddEval3-data-H/MiddEval3/trainingH/ArtL/im1.png"
# feature2 = test.get_feature_vector(imgpath)

# print(np.sum(feature1[12100]*feature2[12100]))
# print(np.sum(feature1[12100]*feature2[12040]))
