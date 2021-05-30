import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

class DataLoader:

    def __init__(self, verbose=False):

        self.verbose = verbose
        
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.y_reg_train = []
        self.y_reg_test = []

        self.timesteps = 0
        
        self.maxs = []
    
    def load_data(self):
    
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        #normalize the images
        x_train = np.expand_dims(x_train/255.0, axis=-1)
        x_test = np.expand_dims(x_test/255.0, axis=-1)
        
        #discretize the images
        x_train = np.where(x_train<0.5, 0, 1)
        x_test = np.where(x_test<0.5, 0, 1)
        
        #create (simulate) a synthetic "time series" data vector (y) for each of the input (x) such that y=Gx and G is linear
        #G represents some abstract function (i.e. fluid flow simulator)
        G = np.load('G.npy')
        
        y_dim = G.shape[-1]
        y_reg_train = np.zeros([y_train.shape[0], y_dim])
        y_reg_test = np.zeros([y_test.shape[0], y_dim])
        
        #simulate Y = GX
        for i in range(y_train.shape[0]):
            y_reg_train[i:i+1, :] = np.reshape((x_train[i:i+1, :, :, 0]), [1, x_train.shape[1]*x_train.shape[2]])@G

        for i in range(y_test.shape[0]):
            y_reg_test[i:i+1, :] = np.reshape((x_test[i:i+1, :, :, 0]), [1, x_test.shape[1]*x_test.shape[2]])@G
            
        #normalize data
        self.maxs = np.max(y_reg_train, axis=0)
        y_reg_train = y_reg_train/self.maxs
        y_reg_test = y_reg_test/self.maxs

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_reg_train = y_reg_train
        self.y_reg_test = y_reg_test
        
        if self.verbose: 
            print("Loaded training data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_train.shape), str(self.y_reg_train.shape), str(self.y_train.shape)))
            print("Loaded testing data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_test.shape), str(self.y_reg_test.shape), str(self.y_test.shape)))
            
        return self.x_train, self.x_test, self.y_train, self.y_test, self.y_reg_train, self.y_reg_test
      
  
             
 