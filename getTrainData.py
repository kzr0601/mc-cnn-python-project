#encoding=utf-8

#change this code to 

#this for read train data
import numpy as np
import os
import cv2
import random

#fetch data in order at the first time, then fectch data randomly
class Data(object):

    def __init__(self, datapath, batchSize, datanum=0, trainable=True):
        dataSetSize = len([name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))])
        if datanum == 0:
            self._num_data = dataSetSize
        else:
            self._num_data = datanum
            
        self._order = np.empty([self._num_data]).astype(np.int32)
        if datanum == 0:
            self._order = np.array([i for i in range(self._num_data)])
        else:
            for i in range(self._num_data):
                self._order[i] = random.randint(0, dataSetSize-1)
        
        random.shuffle(self._order[:self._num_data])
        self._cur = 0
        self._data_path = datapath
        self._batchSize = batchSize

     # def getDataSize(self):
     # 	return self._num_data
    
    def __get_next_minibatch(self):
    	data_L = np.empty((self._batchSize, 11, 11, 1))
    	data_negR = np.empty((self._batchSize, 11, 11, 1))
    	data_posR = np.empty((self._batchSize, 11, 11, 1))
    	label = np.empty(self._batchSize)  #is label necessary?

    	for i in range(self._batchSize):
    		id = self._order[(self._cur+i)%self._num_data]
    		img_L = np.load(self._data_path+'/'+str(id)+'/L.npy')
    		img_negR = np.load(self._data_path+'/'+str(id)+'/negR.npy')
    		img_posR = np.load(self._data_path+'/'+str(id)+'/posR.npy')
    		
    		#cv2.imshow('img_L', img_L)
    		#cv2.waitKey(5000)
    		#print(id)
    		try:
	    		data_L[i, :, :, :] = np.reshape(img_L, (11, 11, 1))
	    		data_negR[i, :, :, :] = np.reshape(img_negR, (11, 11, 1))
	    		data_posR[i, :, :, :] = np.reshape(img_posR, (11, 11, 1))
	    	except BaseException:
	    		print(img_L.shape)
                # print(
                # print(os.path.abspath(self._data_path+'/'+str(id)+'/posR.npy'))
	    	else:
	    		pass

    	if self._cur+self._batchSize >= self._num_data:
    		random.shuffle(self._order[:self._num_data])
    	self._cur = (self._cur+self._batchSize)%self._num_data
    	return {'L': data_L, 'negR': data_negR, 'posR':data_posR}

    def forward(self):
    	batch = self.__get_next_minibatch()
    	return batch


'''
#test
if __name__ == "__main__":
	test = Data("C:/Users/capture/Documents/Visual Studio 2015/Projects/ppm2png/ppm2png/trainData", 1)
	batch = test.forward()
	negR = batch['negR']
	img = np.asarray(negR[0])
	cv2.imshow('img', img)
	cv2.waitKey(5000)
'''