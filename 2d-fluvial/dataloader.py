import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DataLoader:

	def __init__(self, simulator, scenario = 0, verbose=False):

		self.verbose = verbose
		self.scenario = scenario

		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []
		self.y_reg_train = []
		self.y_reg_test = []

		self.sim = simulator
		self.y_min = []
		self.y_max = []
		
	def normalize(self, x):
		x_min = np.min(x)
		x_max = np.max(x)
		return (x - x_min)/(x_max - x_min)
		
	def normalize_data(self, y):
		self.y_min = np.min(y, axis=0)
		self.y_max = np.max(y, axis=0)
		return (y - self.y_min)/(self.y_max - self.y_min)
		
	def normalize_data_(self, y):
		return (y - self.y_min)/(self.y_max - self.y_min)
		
	def load_data(self):
		'''create (simulate) a synthetic "time series" data vector (y) for each of the input (x) such that y=Gx and G is linear
		self.sim  represents some abstract function (i.e. fluid flow simulator)
		'''
		x = np.load("data\M.npy")
		
		#create label, for every 500 models for 5 scenarios
		y = np.zeros([x.shape[0], ], dtype=np.int32)
		for i in range(5):
			y[i*500:i*500+500] = i
		
		#filter by scenario
		x = x[y == self.scenario]
		x = self.normalize(x)
		
		#reshape the models
		x_r = np.zeros([x.shape[0], 100, 100, 1])
		for i in range(x.shape[0]):
			x_r[i,:,:,:] = np.reshape(x[i,:], [1, 100, 100, 1])
		x = x_r
		
		#run forward simulation
		y_reg = self.simulator(x)
		
		#normalize production responses
		y_reg = self.normalize_data(y_reg)
		
		#randomly sample
		np.random.seed(999)
		indexes = np.random.permutation(np.arange(0, x.shape[0], dtype=np.int32))
		partition = int(x.shape[0]*0.8)
		train_idx = indexes[0:partition]
		test_idx = indexes[partition:]

		self.x_train = x[train_idx]
		self.x_test = x[test_idx]

		self.y_reg_train = np.squeeze(y_reg[train_idx])
		self.y_reg_test = np.squeeze(y_reg[test_idx])

		if self.verbose: 
			print("Loaded training data x {:s} and y {:s}".format(str(self.x_train.shape), str(self.y_reg_train.shape)))
			print("Loaded testing data x {:s} and y {:s}".format(str(self.x_test.shape), str(self.y_reg_test.shape)))
		    
		return self.x_train, self.x_test, self.y_reg_train, self.y_reg_test

	def simulator(self, ms):
		'''simulate observations for a given set of models
		'''
		d_dim = self.sim.shape[-1]
		ds = np.zeros([ms.shape[0], d_dim])

		for i in range(ms.shape[0]):
			print("Running simulation ", i)
			ds[i:i+1, :] = np.reshape((ms[i:i+1, :, :, 0]), [1, ms.shape[1]*ms.shape[2]])@self.sim 

		return np.expand_dims(ds, axis=-1)
             
 