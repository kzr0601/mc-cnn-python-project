import numpy as np
import cv2
from enum import Enum
import struct
import os
import datetime
import sys

# class Flag(Enum):
# 	correct = 0
# 	mismatch = 1
# 	occlusion = 2

class Param(object):

	def __init__(self, sgm_P1=1.5, sgm_P2=15, sgm_Q1=4, sgm_Q2=8, sgm_V=1.5, sgm_D=0.08, blur_sigma=6, blur_thre=2):
		self._sgm_P1 = sgm_P1
		self._sgm_P2 = sgm_P2
		self._sgm_Q1 = sgm_Q1
		self._sgm_Q2 = sgm_Q2
		self._sgm_V = sgm_V
		self._sgm_D = sgm_D
		self._blur_sigma = blur_sigma
		self._blur_thre = blur_thre

	def getPara(self):
		return self._sgm_P1, self._sgm_P2, self._sgm_Q1, self._sgm_Q2, self._sgm_V, self._sgm_D, self._blur_sigma, self._blur_thre

class Direction(Enum):
	left = 0
	right = 1
	up = 2
	dowm = 3

class Neighbor(object):

	def __init__(self, o_height, o_width, direction, height, width):
		self._oheight = o_height
		self._owidth = o_width
		self._dir = direction
		self._height = height
		self._width = width

	def isLegal(self):
		if(self._dir == Direction.left):
			return self._owidth>0
		elif(self._dir == Direction.right):
			return self._owidth < self._width-1
		elif(self._dir == Direction.up):
			return self._oheight > 0
		else:
			return self._oheight < self._height-1

	def value(self):
		if(self._dir == Direction.left):
			return self._oheight, self._owidth-1
		elif(self._dir == Direction.right):
			return self._oheight, self._owidth+1
		elif(self._dir == Direction.up):
			return self._oheight-1, self._owidth
		else:
			return self._oheight+1, self._owidth


def semiglobal_matching(img_path, dmax, disp_path, mcv_path, mcv_after_path, disp_after_path):

	# semiglobal matching

	#read param, disp_raw, matching cost volume
	param = Param()
	sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_V, sgm_D, blur_sigma, blur_thre = param.getPara()

	path = img_path #"./MiddEval3-data-H/MiddEval3/trainingH/ArtL"
	# path = "./cal_error_rate/"
	print(os.path.abspath(path+"/im0.png"))
	img = cv2.imread(path+"/im0.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
	imgR = cv2.imread(path+"/im1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
	height, width = img.shape[0], img.shape[1]
	print(str(height)+" "+str(width))
	dmax = dmax #int(input("please input the max disparity: "))

	disp = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE)
	mcv = np.memmap(mcv_path, dtype=np.float32, shape=(1, dmax, height, width))
	mcv = mcv[0]
	mcv2 = np.empty([dmax, height, width])

	start = datetime.datetime.now()

	#SGM
	direction_num = 4
	cost = np.ones((4, dmax, height, width))

	for k in range(4):
		cost[k, :, :, :] = mcv

	for i in range(height):
		for j in range(width):

			for k in [0,2]:

				neighbor = Neighbor(i, j, Direction(k), height, width)
				if not neighbor.isLegal():
					continue
				neighbor_h, neighbor_w = neighbor.value()

				for d in range(dmax):
					P1, P2 = sgm_P1, sgm_P2
					if(j-d >= 0):
						right_w = j-d
						neighborR = Neighbor(i, right_w, Direction(k), height, width)
						if(neighborR.isLegal()):

							#get penalty param
							intensity = img[i][j]
							neighbor_i = img[neighbor_h][neighbor_w]

							neighborR_h, neighborR_w = neighborR.value()
							intensity_R = imgR[i][right_w]
							neighborR_i = imgR[neighborR_h][neighborR_w]

							D1 = abs(intensity - neighbor_i)
							D2 = abs(intensity_R - neighborR_i)

							if(D1<sgm_D and D2<sgm_D):
								P1, P2 = sgm_P1, sgm_P2
							elif(D1>=sgm_D and D2>=sgm_D):
								P1, P2 = sgm_P1/sgm_Q2, sgm_P2/sgm_Q2
							else:
								P1, P2 = sgm_P1/sgm_Q1, sgm_P2/sgm_Q1
							if(Direction(k) == Direction.up or Direction(k) == Direction.dowm):
								P1 = P1 / sgm_V

					if d == 0:
						item1 = cost[k][1][neighbor_h][neighbor_w]+P1
					elif d == dmax-1:
						item1 = cost[k][dmax-2][neighbor_h][neighbor_w]+P1
					else:
						item1 = min(cost[k][d-1][neighbor_h][neighbor_w]+P1, cost[k][d+1][neighbor_h][neighbor_w]+P1)
					item3 = np.min(cost[k, :, neighbor_h, neighbor_w])
					item2 = item3+P2
					cost[k][d][i][j] = mcv[d][i][j] - item3 + min(cost[k][d][neighbor_h][neighbor_w], item1, item2)


	for i in range(height-1, -1, -1):
		for j in range(width-1, -1, -1):

			for k in [1,3]:

				neighbor = Neighbor(i, j, Direction(k), height, width)
				if not neighbor.isLegal():
					continue
				neighbor_h, neighbor_w = neighbor.value()

				for d in range(dmax):
					P1, P2 = sgm_P1, sgm_P2
					if(j-d >= 0):
						right_w = j-d
						neighborR = Neighbor(i, right_w, Direction(k), height, width)
						if(neighborR.isLegal()):

							#get penalty param
							intensity = img[i][j]
							neighbor_i = img[neighbor_h][neighbor_w]

							neighborR_h, neighborR_w = neighborR.value()
							intensity_R = imgR[i][right_w]
							neighborR_i = imgR[neighborR_h][neighborR_w]

							D1 = abs(intensity - neighbor_i)
							D2 = abs(intensity_R - neighborR_i)

							if(D1<sgm_D and D2<sgm_D):
								P1, P2 = sgm_P1, sgm_P2
							elif(D1>=sgm_D and D2>=sgm_D):
								P1, P2 = sgm_P1/sgm_Q2, sgm_P2/sgm_Q2
							else:
								P1, P2 = sgm_P1/sgm_Q1, sgm_P2/sgm_Q1
							if(Direction(k) == Direction.up or Direction(k) == Direction.dowm):
								P1 = P1 / sgm_V

					if d == 0:
						item1 = cost[k][1][neighbor_h][neighbor_w]+P1
					elif d == dmax-1:
						item1 = cost[k][dmax-2][neighbor_h][neighbor_w]+P1
					else:
						item1 = min(cost[k][d-1][neighbor_h][neighbor_w]+P1, cost[k][d+1][neighbor_h][neighbor_w]+P1)
					item3 = np.min(cost[k, :, neighbor_h, neighbor_w])
					item2 = item3+P2
					cost[k][d][i][j] = mcv[d][i][j] - item3 + min(cost[k][d][neighbor_h][neighbor_w], item1, item2)


	#NN-interp... & write acrt & get disparity
	fp = open(mcv_after_path, 'wb')
	for d in range(dmax):
		for i in range(height):
			for j in range(width):
				mcv2[d][i][j] = sum(cost[:, d, i, j])/4
				a = struct.pack('f', mcv2[d][i][j])
				fp.write(a)

	disp2 = np.ones([height, width])
	for r in range(height):
		for c in range(width):
			min_ = 2
			for d in range(dmax):
				if(mcv2[d][r][c] < min_):
					min_ = mcv2[d][r][c]
					disp2[r][c] = d
			# print(d)

	cv2.imwrite(disp_after_path, disp2)

	end = datetime.datetime.now()
	print("semiglobal optimazation uses time: ", end='')
	print(end-start)

def left_right_check(left_disp_path, right_disp_path, disp_path):
	disp_L = cv2.imread(left_disp_path, 0).astype(np.float32)
	disp_R = cv2.imread(right_disp_path, 0).astype(np.float32)

	height = disp_L.shape[0]
	width = disp_L.shape[1]

	flag = np.zeros((height, width)).astype(np.int32)
	for r in range(height):
		for c in range(width):
			d_l = disp_L[r][c]
			d_r = disp_R[r][int(c-d_l)]
			diff = abs(d_l - d_r)
			if diff <= 1:
				flag[r][c] = 0 #Flag.correct
			else:
				tmpflag = True
				for d in range(c):
					if d == int(d_l):
						continue
					d_r = disp_R[r][int(c-d)]
					if abs(d-d_r) <= 1:
						tmpflag = False
						flag[r][c] = 1 #Flag.mismatch
						# disp_L[r][c] = d  #4/11
						break
				if tmpflag:
					flag[r][c] = 2 #Flag.occlusion

	# mismatch_count = 0
	for r in range(height):
		for c in range(width):
			if flag[r][c] != 0:  #4/11
				count = 1
				while c-count >= 0 :
					if flag[r][c-count] == 0: #Flag.correct:
						disp_L[r][c] = disp_L[r][c-count]
						break
					count += 1
			# elif flag[r][c] == 1:
			# 	mismatch_count += 1

	# print(mismatch_count)
	cv2.imwrite(disp_path, disp_L)

def subpixel_enhancement(input_path, output_path, match_cost_volume_path, dmax):
	raw_disp = cv2.imread(input_path, 0)
	height = raw_disp.shape[0]
	width = raw_disp.shape[1]

	vol = np.memmap(match_cost_volume_path, dtype=np.float32, shape=(1, dmax, height, width))
	vol = vol[0]

	disp = np.empty((height, width))
	for r in range(height):
		for c in range(width):
			d = raw_disp[r][c]
			if(d == dmax-1 or d == 0):
				disp[r][c] = d
				continue
			cost_add = vol[d+1][r][c]
			cost_minus = vol[d-1][r][c]
			cost = vol[d][r][c]
			disp[r][c] = d - (cost_add-cost_minus) * 1.0 / (2* (cost_add-2*cost+cost_minus))

	cv2.imwrite(output_path, disp)

def median_filter(input_path, output_path, filter_size = 5):
	raw_disp = cv2.imread(input_path, 0)
	# 5*5 median filter 
	median_disp = cv2.medianBlur(raw_disp, filter_size)
	cv2.imwrite(output_path, median_disp)

def bilateral_filter(input_path, output_path, sigma_color, sigma_space, filter_size=5):
	raw_disp = cv2.imread(input_path, 0)
	bilateral_disp = cv2.bilateralFilter(raw_disp, filter_size, sigma_color, sigma_space)
	cv2.imwrite(output_path, bilateral_disp)


if __name__ == "__main__":

	dmax = int(sys.argv[1])
	name = sys.argv[2]

	path = "../result_10M/"+name
	img_path = "../MiddEval3-data-H/MiddEval3/trainingH/"+name+"/"

	left_disp_path = path+"_l/raw_disp.png"
	right_disp_path = path+"/raw_disp.png"
	left_mcv_path = path+"_l/im0.acrt"
	right_mcv_path = path+"/im0.acrt"

	path = path+"/post"
	if not os.path.exists(path):
		os.makedirs(path)

	left_mcv_after_path = path+"/im0_semi.acrt"
	right_mcv_after_path = path+"/im1_semi.acrt"
	left_disp_after_path = path+"/disp_semi_L.png"
	right_disp_after_path = path+"/disp_semi_R.png"

	left_right_disp_path = path+"/left_right.png"

	disp_subpixel_path = path+"/subpixel.png"
	# dmax = int(input("please input the max possible disparity: "))

	disp_median_path = path+"/median.png"
	disp_bilateral_path = path+"/bilateral.png"

	# hyperparameters from paper
	blur_sigma = 6
	blur_threshold = 2

	print("left semi....."+str(datetime.datetime.now()))
	# one by one
	semiglobal_matching(img_path, dmax, left_disp_path, left_mcv_path, left_mcv_after_path, left_disp_after_path)
	print("right semi....."+str(datetime.datetime.now()))
	semiglobal_matching(img_path, dmax, right_disp_path, right_mcv_path, right_mcv_after_path, right_disp_after_path)
	print("left right check....."+str(datetime.datetime.now()))
	left_right_check(left_disp_after_path, right_disp_after_path, left_right_disp_path)
	print("subpixel enhancement....."+str(datetime.datetime.now()))
	subpixel_enhancement(left_right_disp_path, disp_subpixel_path, left_mcv_after_path, dmax )
	print("filter....."+str(datetime.datetime.now()))
	median_filter(disp_subpixel_path, disp_median_path)
	bilateral_filter(disp_median_path, disp_bilateral_path, sigma_color=blur_threshold, sigma_space=blur_sigma)
	print("end....."+str(datetime.datetime.now()))
