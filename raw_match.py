#L - disp = R

#in model, padding around original img with 5 on each edge

from model import Model
import numpy as np
import cv2
import gc
from pfm import writePfm
import tensorflow as tf
import struct
import os
import sys
from datetime import datetime

# left_or_right = True: choose left as referenced img
# left_or_right = False: choose right as referenced img
def main(imgL_path, imgR_path, model_path ,dmax, acrt_path, raw_disp_path, pfm_path, left_or_right):

	#load model
	my_model = Model(model_path)
	#get img size
	img = cv2.imread(imgL_path, 0)
	height = img.shape[0]
	width = img.shape[1]

	#initialize to be -1
	matching_cost_volume = np.ones([dmax, height, width], dtype='float32')
	matching_cost_volume = -1 * matching_cost_volume
	raw_disparity = np.empty([height, width], dtype='float32')

	#preprocess img
	img_L = cv2.imread(imgL_path, 0).astype("float32")
	height = img_L.shape[0]
	width = img_L.shape[1]
	mean = np.mean(img_L)
	stddev = np.std(img_L)
	# img = img.astype(np.float32)
	img_L = (img_L - mean) / stddev

	# img_pad = np.pad(img, 5, 'constant', constant_values = self._pad_value)
	# edge 'linear_ramp'
	img_L_pad = np.pad(img_L, 5, 'edge')

	#preprocess img
	img_R = cv2.imread(imgR_path, 0).astype("float32")
	height = img_R.shape[0]
	width = img_R.shape[1]
	mean = np.mean(img_R)
	stddev = np.std(img_R)
	img_R = (img_R - mean) / stddev
	img_R_pad = np.pad(img_R, 5, 'edge')

	#compute matching cost upon feature vector using dot mult, using neg matching cost
	for r in range(0, height):
		feature_mat_L = my_model.get_feature_vector(img_L_pad, r)
		feature_mat_R = my_model.get_feature_vector(img_R_pad, r)
		if left_or_right :
			for c in range(0, width):
				max_ = -2
				index = 0
				for d in range(0, min(dmax, c) ):
					#dot mult
					tmp = (feature_mat_L[c] * feature_mat_R[c-d]).astype(np.float32)
					matching_cost_volume[d][r][c] = np.sum(tmp)
					# tmp = tf.cast( tf.multiply(feature_mat_L[num], feature_mat_R[num]), tf.float32 )
					# matching_cost_volume[d][r][c] = tf.reduce_sum( tmp, 0) #tf.reshape(tmp, (1, 64)) , 1 ) 
					if matching_cost_volume[d][r][c] > max_:
						max_ = matching_cost_volume[d][r][c]
						index = d
				# if c < dmax:
				# 	for d in range(c, dmax):
				# 		matching_cost_volume[d][r][c] = -1
				raw_disparity[r][c] = index
		else:
			for c in range(0, width):
				max_ = -2
				index = 0
				for d in range(0, min(dmax, width-c) ):
					#dot mult
					tmp = (feature_mat_L[c+d] * feature_mat_R[c]).astype(np.float32)
					matching_cost_volume[d][r][c] = np.sum(tmp)
					# tmp = tf.cast( tf.multiply(feature_mat_L[num], feature_mat_R[num]), tf.float32 )
					# matching_cost_volume[d][r][c] = tf.reduce_sum( tmp, 0) #tf.reshape(tmp, (1, 64)) , 1 ) 
					if matching_cost_volume[d][r][c] > max_:
						max_ = matching_cost_volume[d][r][c]
						index = d
				# if c < dmax:
				# 	for d in range(c, dmax):
				# 		matching_cost_volume[d][r][c] = -1
				raw_disparity[r][c] = index

	cv2.imwrite(raw_disp_path, raw_disparity)
	writePfm(raw_disparity, pfm_path)

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



if __name__ == "__main__" :
	#dmax is the given maximum possible disparity
	# dmax = int(input("please input the maximum disparity: "))
	# name = input("name: ")

	dmax = int(sys.argv[1])
	name = sys.argv[2]
	model_num = int(sys.argv[3])
	print(name)

	imgL_path = "./MiddEval3-data-H/MiddEval3/trainingH/"+name+"/im0.png"
	imgR_path = "./MiddEval3-data-H/MiddEval3/trainingH/"+name+"/im1.png"

	if(model_num == 1):
		model_path = "./checkpoint_100K_10K"
		path = "./result_100K/"+name+"_l"
	elif(model_num == 2):
		model_path = "./checkpoint_1M_100K"
		path = "./result_1M/"+name+"_l"
	else:
		model_path = "./checkpoint_10M_100K"
		path = "./result_10M/"+name+"_l"

	acrt_path = path+"/im0.acrt"
	raw_disp_path = path+"/raw_disp.png"
	pfm_path = path+"/raw_disp.pfm"

	if not os.path.exists(path):
		os.makedirs(path)

	fp = open(path+"/time.txt", "w")
	start = datetime.now()
	fp.write(str("start: ")+str(start))	
	main( imgL_path, imgR_path, model_path, dmax, acrt_path, raw_disp_path, pfm_path, True)
	end = datetime.now()
	fp.write(str("end: ")+str(end))
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
