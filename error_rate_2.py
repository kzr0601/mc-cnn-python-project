# evaluate on error rate
# all

from pfm import *
import cv2
import numpy as np
import sys
import os
import math


# name = sys.argv[1]
name = input("please input name: ")
disp0GT_00 = readPfm("../MiddEval3-GT0-H/MiddEval3/trainingH/"+name+"/disp0GT.pfm")

disp0_names = []
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\bilateral.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\whole.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\no_semi.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\no_left-right.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\no_subpixel.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\no_median.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\no_bilateral.png")
disp0_names.append("D:\\kzr\\mc_cnn_project\\result_10M\\"+name+"\\post\\median.png")

for disp0_name in disp0_names:
	
	disp0 = cv2.imread(disp0_name, 0)

	print("disp0_name: "+disp0_name)
	# disp0 = readPfm("./result_"+name+"/raw_disp.pfm")
	height = disp0.shape[0]
	width = disp0.shape[1]

	threshold = 1 #int(input("please input threshold: "))
	error = np.abs(disp0GT_00-disp0)
	error_all = error > threshold
	print("all: "+str(np.sum(error_all == True) * 1.0 / (disp0.shape[0]*disp0.shape[1]) ) )

	masknonocc = cv2.imread("../MiddEval3-GT0-H/MiddEval3/trainingH/"+name+"/mask0nocc.png", 0)
	# masknonocc = cv2.imread("./nonocc_small.png", 0)
	masknonocc = masknonocc == 255

	disp_noninf = np.isinf(disp0GT_00)
	disp_noninf = disp_noninf == False

	mask = masknonocc & disp_noninf
	nonocc = np.sum(mask == True)

	error_nonocc = error_all & mask
	print("nonocc: "+ str(np.sum(error_nonocc == True) * 1.0 / nonocc) )
	print("=========================================================")
