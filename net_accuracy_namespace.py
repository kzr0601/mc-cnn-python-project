
#encoding=utf-8
import tensorflow as tf
import numpy as np
from getTrainData_accuracy import Data
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
learning_rate = 0.003
batchSize = 128
epoch = 14
#dataSize = 38,000,000 in paper
dataSize = 1000000
validate_data_size = 100000
_num_iteration = math.ceil(dataSize / batchSize )
_num_batchSize = batchSize
_num_validate_iteration = math.ceil(validate_data_size/batchSize)

#log file 
log = open('log.txt', 'w')
#all file paths
checkpointPath = "./checkpoint_accuracy_100w_10w_namespace"
# dataPath = "./dataSets"
# validate_data_path = "./valid_dataSets_11W"
dataPath="./dataSets_1299118"
validate_data_path="./datasets_10w"

if not os.path.exists(checkpointPath):
	os.makedirs(checkpointPath)

def calculate_train_acc(acc_pos, acc_neg, batchSize):
	acc_pos = acc_pos < 0.5
	acc_neg = acc_neg > 0.5
	acc = (np.sum(acc_pos == True) + np.sum(acc_neg == True) ) * 1.0 / batchSize
	return acc

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
x_L = tf.placeholder("float32", shape=[None, 11, 11])
x_R = tf.placeholder("float32", shape = [None, 11, 11])
x_neg = tf.placeholder("float32", shape = [None, 11, 11])

x_L = tf.reshape(x_L, [-1, 11, 11, 1])
x_R = tf.reshape(x_R, [-1, 11, 11, 1])
x_neg = tf.reshape(x_neg, [-1, 11, 11, 1])

#the first layer
with tf.name_scope('conv1'):
	w_conv1=weight_variable([3, 3, 1, 112])
	b_conv1=bias_variable([112])
	h_conv1=tf.nn.relu(conv2d(x_L, w_conv1)+b_conv1)
	h_conv1_R = tf.nn.relu(conv2d(x_R, w_conv1)+b_conv1)
	h_conv1_neg = tf.nn.relu(conv2d(x_neg, w_conv1)+b_conv1)

#the second layer
with tf.name_scope('conv2'):
	w_conv2=weight_variable([3,3,112,112])
	b_conv2=bias_variable([112])
	h_conv2=tf.nn.relu(conv2d(h_conv1, w_conv2)+b_conv2)
	h_conv2_R = tf.nn.relu(conv2d(h_conv1_R, w_conv2)+b_conv2)
	h_conv2_neg = tf.nn.relu(conv2d(h_conv1_neg, w_conv2)+b_conv2)

#the third layer
with tf.name_scope('conv3'):
	w_conv3=weight_variable([3,3,112,112])
	b_conv3=bias_variable([112])
	h_conv3=tf.nn.relu(conv2d(h_conv2, w_conv3)+b_conv3)
	h_conv3_R=tf.nn.relu(conv2d(h_conv2_R, w_conv3)+b_conv3)
	h_conv3_neg=tf.nn.relu(conv2d(h_conv2_neg, w_conv3)+b_conv3)

#the fourth layer
with tf.name_scope('conv4'):
	w_conv4=weight_variable([3,3,112,112])
	b_conv4=bias_variable([112])
	h_conv4=tf.nn.relu(conv2d(h_conv3, w_conv4)+b_conv4)
	h_conv4_R = tf.nn.relu(conv2d(h_conv3_R, w_conv4)+b_conv4)
	h_conv4_neg = tf.nn.relu(conv2d(h_conv3_neg, w_conv4)+b_conv4)

#the fifth layer, without relu
with tf.name_scope('conv5'):
	w_conv5=weight_variable([3,3,112,112])
	b_conv5=bias_variable([112])
	h_conv5=tf.nn.relu(conv2d(h_conv4, w_conv5)+b_conv5)
	h_conv5_R = tf.nn.relu(conv2d(h_conv4_R, w_conv5)+b_conv5)
	h_conv5_neg = tf.nn.relu(conv2d(h_conv4_neg, w_conv5)+b_conv5)

#concatenate
h_conv5 = tf.reshape(h_conv5, [-1, 112])
h_conv5_R = tf.reshape(h_conv5_R, [-1, 112])
h_conv5_neg = tf.reshape(h_conv5_neg, [-1, 112])
concat_vector = tf.concat([h_conv5, h_conv5_R], 1)
concat_vector_neg = tf.concat([h_conv5, h_conv5_neg], 1)

#the first fully-connected layer
with tf.name_scope('fc1'):
	w_fc1 = weight_variable([224, 384])
	b_fc1 = bias_variable([384])
	h_fc1 = tf.nn.relu(tf.matmul(concat_vector, w_fc1)+b_fc1)
	h_fc1_neg = tf.nn.relu(tf.matmul(concat_vector_neg, w_fc1)+b_fc1)
# #dropout
# keep_prob_1 = tf.placeholder(tf.float32)
# h_fc1 = tf.nn.dropout(h_fc1, keep_prob_1)

#the second fully-connected layer
with tf.name_scope('fc2'):
	w_fc2 = weight_variable([384, 384])
	b_fc2 = bias_variable([384])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
	h_fc2_neg = tf.nn.relu(tf.matmul(h_fc1_neg, w_fc2)+b_fc2)
# #dropout
# keep_prob_2 = tf.placeholder(tf.float32)
# h_fc2 = tf.nn.dropout(h_fc2, keep_prob_2)

#the third fully-connected layer
with tf.name_scope('fc3'):
	w_fc3 = weight_variable([384, 1])
	b_fc3 = bias_variable([1])
	h_fc3 = tf.matmul(h_fc2, w_fc3)+b_fc3
	h_fc3_neg = tf.matmul(h_fc2_neg, w_fc3)+b_fc3

output_score = tf.sigmoid(h_fc3)
output_score_neg = tf.sigmoid(h_fc3_neg)

# #binary cross-entropy loss
# loss = tf.log(output_score) + tf.log(1-output_score_neg) 
# loss = tf.reduce_mean(loss)
# acc = output_score_neg - output_score

# 梯度丢失问题 ？ , 因为h_fc3总是[0, 0, ..., 0], 
# 梯度丢失问题可以用sigmoid 解决
# above code 如果output_score很小很小，loss会趋于-inf
# 可是为什么loss总是nan, acc总是0， 在epoch>1以后

const = tf.ones(shape=[batchSize], dtype=tf.float32) * margin
loss = tf.reduce_mean(tf.maximum(0.0, const- output_score + output_score_neg))
acc = output_score - output_score_neg


#optimizer
# changable learning rate lr = 0.003, decrease by a factor of 10 on the 11th epoch as in paper
lr = tf.placeholder(tf.float32)
train_step = tf.train.MomentumOptimizer(lr, momentum_term).minimize(loss)

print("here")
# sess = tf.InteractiveSession()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)))

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
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

# pos_label = np.ones([int(batchSize/2)])
# neg_label = np.zeros([int(batchSize/2)])

for run in range(epoch):
	now = datetime.datetime.now()
	print(str(run+1)+" epoch number: "+str(now-start))
	#decrease learning rate
	if run > 9:
		_learning_rate = _learning_rate / 10
	for i in range(_num_iteration):
		batch = data.forward()
		if(i%10 == 0):
			train_acc = acc.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess)
			train_acc = train_acc > 0
			train_acc = np.sum(train_acc == True)*1.0/_num_batchSize
			accArray.append(train_acc)

			train_loss = loss.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess)
			print("epoch %d, step %d, loss %g, acc %g"%(run, i, train_loss, train_acc))
			log.write("epoch %d, step %d, loss %g\n, acc %g"%(run, i, train_loss, train_acc) )
			lossArray.append(train_loss)
		train_step.run(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess)
		# print(h_fc3.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess) )
		# print("==============================================================")
		# print(output_score.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess))
		

	#save checkpoint for each epoch
	print("{} Saving checkpoint of model ".format(datetime.datetime.now()) + str(run))  
	checkpoint_name = os.path.join(checkpointPath, 'model_epoch'+str(epoch+1)+'.ckpt')
	saver.save(sess, checkpoint_name)

	#validate model for each epoch
	val_loss_mean = 0
	val_acc_mean = 0
	for i in range(_num_validate_iteration):
		validate_batch = validate_data.forward()
		valid_loss = loss.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess)
		val_loss_mean += valid_loss

		valid_acc = acc.eval(feed_dict={x_L:batch['L'], x_R : batch['posR'], x_neg: batch['negR'], lr: _learning_rate}, session=sess)
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