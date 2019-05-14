#start from line 91 ------  2019.2.28

#encoding=utf-8
import tensorflow as tf
import numpy as np
from getTrainData import Data
import cv2
import os
import math
import matplotlib.pyplot as plt
import datetime
import gc
import struct
#import data here

#hyper parameters
margin = 0.2
momentum_term = 0.9
learning_rate = 0.002
batchSize = 128
epoch = 14
#dataSize = 38,000,000 in paper
dataSize = 38000000 #1000000
validate_data_size = 1000000  #100000
_num_iteration = math.ceil(dataSize / batchSize )
_num_batchSize = batchSize
_num_validate_iteration = math.ceil(validate_data_size/batchSize)

#log file 
log = open('log.txt', 'w')
#all file paths
dataPath = "./dataSets_38M_"
print(os.path.abspath(dataPath))
validate_data_path = "./dataSets_1299118"
checkpointPath = "./checkpoint_38M_1M"
# dataPath = "/home/supergp/kzr/dataSets_1299118"
# checkpointPath = "/home/supergp/kzr/checkpoint"
# validate_data_path = "/home/supergp/kzr/dataSets"

if not os.path.exists(checkpointPath):
	os.makedirs(checkpointPath)

def draw_pic(array, str):
	plt.figure()
	x = np.array(range(len(array)))
	array = np.array(array)
	plt.plot(x, array)
	plt.savefig(str+".png")

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

#here, None is batchSize
x = tf.placeholder("float32", shape=[None, 11, 11])
x_neg = tf.placeholder("float32", shape = [None, 11, 11])
x_pos = tf.placeholder("float32", shape = [None, 11, 11])

x = tf.reshape(x, [-1, 11, 11, 1])
x_neg = tf.reshape(x_neg, [-1, 11, 11, 1])
x_pos = tf.reshape(x_pos, [-1, 11, 11, 1])

#the first layer
w_conv1=weight_variable([3, 3, 1, 64])
b_conv1=bias_variable([64])
h_conv1=tf.nn.relu(conv2d(x, w_conv1)+b_conv1)
h_conv1_neg = tf.nn.relu(conv2d(x_neg, w_conv1)+b_conv1)
h_conv1_pos = tf.nn.relu(conv2d(x_pos, w_conv1)+b_conv1)

#the second layer
w_conv2=weight_variable([3,3,64,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_conv1, w_conv2)+b_conv2)
h_conv2_neg = tf.nn.relu(conv2d(h_conv1_neg, w_conv2)+b_conv2)
h_conv2_pos = tf.nn.relu(conv2d(h_conv1_pos, w_conv2)+b_conv2)

#the third layer
w_conv3=weight_variable([3,3,64,64])
b_conv3=bias_variable([64])
h_conv3=tf.nn.relu(conv2d(h_conv2, w_conv3)+b_conv3)
h_conv3_neg=tf.nn.relu(conv2d(h_conv2_neg, w_conv3)+b_conv3)
h_conv3_pos=tf.nn.relu(conv2d(h_conv2_pos, w_conv3)+b_conv3)

#the fourth layer
w_conv4=weight_variable([3,3,64,64])
b_conv4=bias_variable([64])
h_conv4=tf.nn.relu(conv2d(h_conv3, w_conv4)+b_conv4)
h_conv4_neg = tf.nn.relu(conv2d(h_conv3_neg, w_conv4)+b_conv4)
h_conv4_pos = tf.nn.relu(conv2d(h_conv3_pos, w_conv4)+b_conv4)

#the fifth layer, without relu
w_conv5=weight_variable([3,3,64,64])
b_conv5=bias_variable([64])
h_conv5=conv2d(h_conv4, w_conv5)+b_conv5
h_conv5_neg = conv2d(h_conv4_neg, w_conv5)+b_conv5
h_conv5_pos = conv2d(h_conv4_pos, w_conv5)+b_conv5

#normalize
h_conv5 = tf.reshape(h_conv5, [-1, 64])
h_conv5_neg = tf.reshape(h_conv5_neg, [-1, 64])
h_conv5_pos = tf.reshape(h_conv5_pos, [-1, 64])
vec_L = tf.nn.l2_normalize(h_conv5, axis=1)  #dim -> axis
vec_neg = tf.nn.l2_normalize(h_conv5_neg, axis=1)
vec_pos = tf.nn.l2_normalize(h_conv5_pos, axis=1)
#dot product
tmp_pos = tf.cast(tf.multiply(vec_L, vec_pos), tf.float32)
pos = tf.reduce_sum(tmp_pos, 1)
tmp_neg = tf.cast(tf.multiply(vec_L, vec_neg), tf.float32)
neg = tf.reduce_sum(tmp_neg, 1)
const = tf.ones(shape=[batchSize], dtype=tf.float32) * margin
#hindge loss = max(0, 0.2+neg-pos)
loss = tf.reduce_mean(tf.maximum(0.0, const-pos+neg))
acc = pos-neg

#optimizer
# changable learning rate lr = 0.002, decrease by a factor of 10 on the 11th epoch as in paper
lr = tf.placeholder(tf.float32)
train_step = tf.train.MomentumOptimizer(lr, momentum_term).minimize(loss)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

print("here")
sess = tf.Session()#config=tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)))
sess.run(tf.global_variables_initializer())

data = Data(dataPath, _num_batchSize, dataSize)
validate_data = Data(validate_data_path, _num_batchSize, validate_data_size)

lossArray = []
validLossArray = []
accArray = []
validAccArray = []
saver = tf.train.Saver()

start = datetime.datetime.now()
_learning_rate = learning_rate
print("epoch: "+str(epoch))
for run in range(epoch):
	now = datetime.datetime.now()
	print(str(run+1)+" epoch number: "+str(now-start))
	#decrease learning rate
	if run > 9:
		_learning_rate = _learning_rate / 10
	for i in range(_num_iteration):
		batch = data.forward()
		if(i%1 == 0):
			train_acc = acc.eval(feed_dict={x:batch['L'], x_neg : batch['negR'], x_pos : batch['posR'], lr: _learning_rate}, session=sess)
			train_acc = train_acc > 0
			train_acc = np.sum(train_acc == True)*1.0/_num_batchSize
			accArray.append(train_acc)

			train_loss = loss.eval(feed_dict={x:batch['L'], x_neg : batch['negR'], x_pos : batch['posR'], lr: _learning_rate}, session=sess)
			print("epoch %d, step %d, loss %g, acc %g"%(run, i, train_loss, train_acc))
			log.write("epoch %d, step %d, loss %g\n, acc %g"%(run, i, train_loss, train_acc) )
			lossArray.append(train_loss)
		train_step.run(feed_dict={x:batch['L'], x_neg : batch['negR'], x_pos : batch['posR'], lr: _learning_rate}, session=sess)

	#save checkpoint for each epoch
	print("{} Saving checkpoint of model ".format(datetime.datetime.now()) + str(run))  
	checkpoint_name = os.path.join(checkpointPath, 'model_epoch'+str(epoch+1)+'.ckpt')
	saver.save(sess, checkpoint_name)

	#validate model for each epoch
	val_loss_mean = 0
	val_acc_mean = 0
	for i in range(_num_validate_iteration):
		validate_batch = validate_data.forward()
		validate_loss = loss.eval(feed_dict={x:validate_batch['L'], x_neg : validate_batch['negR'], x_pos : validate_batch['posR'], lr: _learning_rate}, session=sess)
		val_loss_mean += validate_loss

		valid_acc = acc.eval(feed_dict={x:batch['L'], x_neg : batch['negR'], x_pos : batch['posR'], lr: _learning_rate}, session=sess)
		valid_acc = valid_acc > 0
		valid_acc = np.sum(valid_acc == True)*1.0/_num_batchSize
		val_acc_mean += valid_acc
	val_loss_mean = val_loss_mean * 1.0 / _num_validate_iteration
	val_acc_mean = val_acc_mean * 1.0 / _num_validate_iteration
	print("epoch %d, validation loss %g, valid acc %g"%(run, val_loss_mean, val_acc_mean))
	log.write("epoch %d, validation loss %g, valid acc %g\n"%(run, val_loss_mean, val_acc_mean))
	validLossArray.append(val_loss_mean)
	validAccArray.append(val_acc_mean)


end = datetime.datetime.now()
print("time of trainning: " + str(end-start))

draw_pic(lossArray, "train_loss")
draw_pic(validLossArray, "valid_loss")
draw_pic(accArray, "train_acc")
draw_pic(validAccArray, "valid_acc")

print("end")