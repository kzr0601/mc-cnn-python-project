#get data sets, neg sample + pos sample
import os
import numpy as np
import cv2
import random
import struct
import sys
from pfm import *

#hyper_param
dataset_neg_low = 1.5
dataset_neg_high = 6
dataset_pos = 0.5 #0.5

def preprocess_pic_with_edge_pad(img, pad_length=5):
	img0 = np.pad(img, pad_length, 'edge')
	mean = np.mean(img0)
	stddev = np.std(img0)
	img0 = img0.astype(np.float32)
	img0 = (img0 - mean) / stddev
	return img0

def getPatch(img, patch_count, width):
	r = int(patch_count / width)
	c = int(patch_count % width)
	p = img[r: r+11]
	patch = p[:, range(c, c+11)]
	return patch

path = "./dataSets_38M_"
if not os.path.exists(path):
	os.makedirs(path)
print(os.path.abspath(path))
img_path = "./MiddEval3-data-H/MiddEval3/trainingH/"
gt_path = "./MiddEval3-GT0-H/MiddEval3/trainingH/"
file_names = os.listdir(img_path)

#how many samples we have already
samples = os.listdir(path)
count = len(samples)
print(count)

#which img is the next sample came from
file_count = int(sys.argv[1]) #(int)(input("please input file_count: "))
patch_count = int(sys.argv[2]) #(int)(input("please input patch_count: "))
# expected sample number
up_bound = int(sys.argv[3]) #int(input("please input expected sample number: "))

flag = 0
while count < up_bound:

	if flag == 0:
		#get the size of current img
		L_img = cv2.imread(img_path+file_names[file_count]+"/im0.png", 0)
		R_img = cv2.imread(img_path+file_names[file_count]+"/im1.png", 0)
		height = L_img.shape[0]
		width = L_img.shape[1]
		print(str(height)+" "+str(width))

		disp0GT = readPfm(gt_path+file_names[file_count]+"./disp0GT.pfm")
		print(str(disp0GT.shape[0])+" "+str(disp0GT.shape[1]))

		L = preprocess_pic_with_edge_pad(L_img)
		R = preprocess_pic_with_edge_pad(R_img)

		flag = 1

	#get disp0 groundtruth
	# r = int(patch_count / (width-10) ) + 5
	# c = int(patch_count % (width-10) ) + 5
	r = int(patch_count / width)
	c = int(patch_count % width)
	if r >= height or c >= width:
		patch_count = 0
		file_count = file_count + 1
		flag = 0
		continue
	disp = disp0GT[r][c]
	#exclude pixels with disparity inf
	if disp == float('inf') or c - disp < 0:
		patch_count = patch_count + 1
		#go through all the patches in this file
		if patch_count == height*width-1:
			patch_count = 0
			file_count = file_count + 1
			flag = 0
		#go through all the files
		if file_count == 15:
			break
		continue

	#negR
	o_neg = dataset_neg_low + (dataset_neg_high- dataset_neg_low) * random.random() 
	if round(c - disp - o_neg) < 0:
		negR_count = round(patch_count - disp + o_neg)
	elif round(c -disp + o_neg) >= width:
		negR_count = round(patch_count - disp - o_neg)
	else:
		f = random.randint(1, 4)
		if f%2 == 0:
			flag = 1
		else:
			flag = -1
		negR_count = round(patch_count -disp + flag*o_neg)

	#posR
	o_pos = dataset_pos-random.random()*dataset_pos*2
	if round(c - disp + o_pos) < 5:
		posR_count =  round(patch_count - disp - o_pos)
	elif round(c - disp +o_pos) > width-6:
		posR_count =  round(patch_count - disp - o_pos)
	else:
		posR_count =  round(patch_count - disp + o_pos)


	negR = getPatch(R, negR_count, width)
	posR = getPatch(R, posR_count, width)
	L_pacth = getPatch(L, patch_count, width)

	#store
	if not os.path.exists(path+"/"+str(count)):
		os.makedirs(path+"/"+str(count))
	np.save(path+"/"+str(count)+"/L.npy", L_pacth)
	np.save(path+"/"+str(count)+"/negR.npy", negR)
	np.save(path+"/"+str(count)+"/posR.npy", posR)

	#count++
	count = count + 1
	patch_count = patch_count + 1
	#go through all the patches in this file
	if patch_count == height*width-1:
		patch_count = 0
		file_count = file_count + 1
		flag = 0
	#go through all the files
	if file_count == 15:
		break

#store file_count and patch_count for next time
f1 = open('count.txt', 'w')
f1.write(str(file_count)+" "+str(patch_count))
f1.close()

