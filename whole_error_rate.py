# evaluate on error rate
# all

from pfm import *
import cv2
import numpy as np
import sys
import os
import math


file_names = ["Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
all_num = 15
train_names = ["ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves"]
train_num = 12
valid_names = ["Adirondack", "Teddy", "Vintage"]
valid_num = 3

error_all_all = 0
error_nonocc_all = 0

for name in file_names:
	disp0GT_00 = readPfm("../MiddEval3-GT0-H/MiddEval3/trainingH/"+name+"/disp0GT.pfm")
	disp0_name = "D:\\kzr\\mc_cnn_project\\result_1M\\"+name+"_l\\raw_disp.png"
	# disp0_name = "D:\\kzr\\mc_cnn_project\\result\\result_"+name+"\\disp0.png"
	disp0 = cv2.imread(disp0_name, 0)

	height = disp0.shape[0]
	width = disp0.shape[1]

	threshold = 1
	error = np.abs(disp0GT_00-disp0)
	error_all = error > threshold
	error_all_all += np.sum(error_all == True) * 1.0 / (disp0.shape[0]*disp0.shape[1]) 

	masknonocc = cv2.imread("../MiddEval3-GT0-H/MiddEval3/trainingH/"+name+"/mask0nocc.png", 0)
	# masknonocc = cv2.imread("./nonocc_small.png", 0)
	masknonocc = masknonocc == 255

	disp_noninf = np.isinf(disp0GT_00)
	disp_noninf = disp_noninf == False

	mask = masknonocc & disp_noninf
	nonocc = np.sum(mask == True)

	error_nonocc = error_all & mask
	error_nonocc_all += np.sum(error_nonocc == True) * 1.0 / nonocc

print("all: "+str(error_all_all/all_num))
print("nonocc: "+str(error_nonocc_all/all_num))


