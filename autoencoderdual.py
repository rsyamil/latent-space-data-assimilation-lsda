import numpy as np
import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv1D, UpSampling1D
from keras.layers import AveragePooling1D, MaxPooling1D

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History, EarlyStopping

from keras.utils import plot_model
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.cm as cm
from matplotlib.colors import Normalize

def RMSE(x, y):
    return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))

def sampling(args):
    
    epsilon_std = 1.0
    
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch, dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon
    
class Autoencoder:

    def __init__(self, M, D, zm_dim=64, zd_dim=16, variational=False, name=[]):
        self.name = name
        self.field = name
        
        self.M = M 
        self.D = D
        
        self.mx_sz = M.shape[1]
        self.my_sz = M.shape[2]
        self.mz_sz = M.shape[3]
        
        self.dx_sz = D.shape[1]
        
        self.zm_dim = zm_dim
        self.zd_dim = zd_dim
        
        self.variational = variational
        
        self.m2m = []
        self.m2zm = []
        self.zm2m = []
        
        self.d2d = []
        self.d2zd = []
        self.zd2d = []
        
        self.zd2zm = []
        self.d2m = []
        
        self.lambdaL1 = 1e-10
        
    def encoder2D(self):
        #define the simple autoencoder
        input_image = Input(shape=(self.mx_sz, self.my_sz, self.mz_sz)) 

        #image encoder
        _ = Conv2D(4, (3, 3), padding='same', name='enc')(input_image)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(8, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(16, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Flatten()(_)

        if not self.variational:
            encoded_image = Dense(self.zm_dim)(_)
        else:
            _ = Dense(self.zm_dim)(_)
            z_mean_m = Dense(self.zm_dim)(_)
            z_log_var_m = Dense(self.zm_dim)(_)
            encoded_image = Lambda(sampling)([z_mean_m, z_log_var_m])
            return input_image, encoded_image, z_mean_m, z_log_var_m

        return input_image, encoded_image

    def decoder2D(self, encoded_image):
        #image decoder
        _ = Dense((256))(encoded_image)
        _ = Reshape((4, 4, 16))(_)

        _ = Conv2D(16, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(8, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(4, (3, 3))(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        decoded_image = Conv2D(1, (3, 3), padding='same')(_)

        return decoded_image
    
    def regressor(self, encoded_data):
        #generalize this later
        
        _ = Dense(20)(encoded_data)
        _ = LeakyReLU(alpha=0.3)(_)
        
        _ = Dense(30)(_)
        _ = LeakyReLU(alpha=0.3)(_)
        
        _ = Dense(40)(_)
        _ = LeakyReLU(alpha=0.3)(_)
        
        _ = Dense(50)(_)
        _ = LeakyReLU(alpha=0.3)(_)
        
        encoded_m_reg = Dense(self.zm_dim)(_)

        return encoded_m_reg
    
    def encoder1D(self):
        
        input_dt = Input(shape=(self.dx_sz,))
        
        _ = Dense(100)(input_dt)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(80)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(60)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(40)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        if not self.variational:
            encoded_d = Dense(self.zd_dim)(_)
        else:
            _ = Dense(self.zd_dim)(_)
            z_mean_d = Dense(self.zd_dim)(_)
            z_log_var_d = Dense(self.zd_dim)(_)
            encoded_d = Lambda(sampling)([z_mean_d, z_log_var_d])
            return input_dt, encoded_d, z_mean_d, z_log_var_d

        return input_dt, encoded_d
    
    def decoder1D(self, encoded_d):
        
        _ = Dense(40)(encoded_d)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(60)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(80)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(100)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        decoded_d = Dense(self.dx_sz)(_)

        return decoded_d
    
    def train_autoencoder_dual(self, epoch=300, load=False):
        
        #data autoencoder
        input_dt, encoded_d = self.encoder1D()
        decoded_d = self.decoder1D(encoded_d)
        #regressor 
        encoded_m_reg = self.regressor(encoded_d)
        #model autoencoder
        decoded_m_reg = self.decoder2D(encoded_m_reg)
        
        self.d2m = Model(input_dt, [decoded_d, decoded_m_reg])
        opt = keras.optimizers.Adam(lr=1e-3)
        self.d2m.compile(optimizer=opt, loss='mean_squared_error')
        self.d2m.summary()
        plot_model(self.d2m, to_file='d2m.png')
        
        #second model autoencoder as a placeholder (soft conditioning)
        input_image, encoded_image = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)
        
        self.m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.m2m.compile(optimizer=opt, loss='mean_squared_error')
        self.m2m.summary()
        plot_model(self.m2m, to_file='m2m.png')
        
        #train the neural network alternatingly
        totalEpoch = epoch
        plot_losses1 = util.PlotLosses()
        plot_losses2 = util.PlotLosses()
        history1 = History()
        history2 = History()
        
        AE_reg = np.zeros([totalEpoch, 6])
        AE_m = np.zeros([totalEpoch, 2])
        
        d2m_idxs = [19, 20, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
        m2m_idxs = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        
        if not load:
            for i in range(totalEpoch):
                #train main reg model
                self.d2m.fit(self.D, [self.D, self.M],        
                            epochs=1,
                            batch_size=128,
                            shuffle=True,
                            validation_split=0.2,
                            callbacks=[plot_losses1, EarlyStopping(monitor='loss', patience=60), history1])
            
                #copy loss
                AE_reg[i, :] = np.squeeze(np.asarray(list(history1.history.values())))
                                
                #copy weights for model decoder only into the second model
                for j in range(len(d2m_idxs)):
                    self.m2m.layers[m2m_idxs[j]].set_weights(self.d2m.layers[d2m_idxs[j]].get_weights())

                #train placeholder model AE
                self.m2m.fit(self.M, self.M,        
                            epochs=1,
                            batch_size=128,
                            shuffle=True,
                            validation_split=0.2,
                            callbacks=[plot_losses2, EarlyStopping(monitor='loss', patience=60), history2])

                #copy into the main model 
                for j in range(len(d2m_idxs)):
                    self.d2m.layers[d2m_idxs[j]].set_weights(self.m2m.layers[m2m_idxs[j]].get_weights())

                #copy loss
                AE_m[i, :] = np.squeeze(np.asarray(list(history2.history.values())))
 
                #write to folder for every 10th epoch for monitoring
                figs = util.plotAllLosses(AE_reg, AE_m)
                figs.savefig('Dual_Losses.png')
                
            #save trained model
            self.d2m.save('d2m.h5')
            self.m2m.save('m2m.h5')
        else:
            #load an already trained model
            print("Trained model loaded")
            self.d2m = load_model('d2m.h5')
            self.m2m = load_model('m2m.h5')
            
        #set all functions (copy weights from layers, incase comp. graph dont exist)
        
        m_f = Input(shape=(self.mx_sz, self.my_sz, self.mz_sz))
        _ = self.m2m.layers[1](m_f)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            _ = self.m2m.layers[i](_)
        zm_f = self.m2m.layers[13](_)
        self.m2zm = Model(m_f, zm_f)
        
        zm_f_2 = Input(shape=(self.zm_dim, )) 
        _ = self.m2m.layers[14](zm_f_2)
        for i in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
            _ = self.m2m.layers[i](_)
        m_f_2 = self.m2m.layers[27](_)
        self.zm2m = Model(zm_f_2, m_f_2)
        
        d_f = Input(shape=(self.dx_sz,))
        _ = self.d2m.layers[1](d_f)
        for i in [2, 3, 4, 5, 6, 7, 8]:
            _ = self.d2m.layers[i](_)
        zd_f = self.d2m.layers[9](_)
        self.d2zd = Model(d_f, zd_f)
        
        zd_f_2 = Input(shape=(self.zd_dim, ))
        _ = self.d2m.layers[24](zd_f_2)
        for i in [26, 28, 30, 32, 34, 36, 38]:
            _ = self.d2m.layers[i](_)
        d_f_2 = self.d2m.layers[40](_)
        self.zd2d = Model(zd_f_2, d_f_2)
        
        d_f_3 = Input(shape=(self.dx_sz,))
        _ = self.d2m.layers[1](d_f_3)
        for i in [2, 3, 4, 5, 6, 7, 8]:
            _ = self.d2m.layers[i](_)
        zd_f_3 = self.d2m.layers[9](_)
        _ = self.d2m.layers[24](zd_f_3)
        for i in [26, 28, 30, 32, 34, 36, 38]:
            _ = self.d2m.layers[i](_)
        d_f_4 = self.d2m.layers[40](_)
        self.d2d = Model(d_f_3, d_f_4)
        
        zd_f_5 = Input(shape=(self.zd_dim, ))
        _ = self.d2m.layers[10](zd_f_5)
        for i in [11, 12, 13, 14, 15, 16, 17]:
            _ = self.d2m.layers[i](_)
        zm_f_3 = self.d2m.layers[18](_)
        self.zd2zm = Model(zd_f_5, zm_f_3)
        
    def train_autoencoder_dual_var(self, epoch=300, load=False):
        
        #data autoencoder
        input_dt, encoded_d, zd_mean, zd_log_var = self.encoder1D()
        decoded_d = self.decoder1D(encoded_d)
        #regressor 
        encoded_m_reg = self.regressor(encoded_d)
        #model autoencoder
        decoded_m_reg = self.decoder2D(encoded_m_reg)
        
        #define the variational loss and mse loss (equal weighting)
        def dvae_loss(input_dt, decoded_d):
            recons_loss = K.sum(mse(input_dt, decoded_d))                
            kl_loss = (- 0.5) * K.sum(1 + zd_log_var - K.square(zd_mean) - K.exp(zd_log_var), axis=-1)
            return K.mean(recons_loss + kl_loss)
        
        #add custom loss 
        get_custom_objects().update({"dvae_loss": dvae_loss})
        
        self.d2m = Model(input_dt, [decoded_d, decoded_m_reg])
        opt = keras.optimizers.Adam(lr=1e-3)
        self.d2m.compile(optimizer=opt, loss=dvae_loss)
        self.d2m.summary()
        plot_model(self.d2m, to_file='d2m_var.png')
        
        #second model autoencoder as a placeholder (soft conditioning)
        input_image, encoded_image, z_mean_m, z_log_var_m = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)
        
        #define the variational loss and mse loss (equal weighting)
        def mvae_loss(input_image, decoded_image):
            recons_loss = K.sum(mse(input_image, decoded_image))                
            kl_loss = (- 0.5) * K.sum(1 + z_log_var_m - K.square(z_mean_m) - K.exp(z_log_var_m), axis=-1)
            return K.mean(recons_loss + kl_loss)
            
        #add custom loss 
        get_custom_objects().update({"mvae_loss": mvae_loss})
        
        self.m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.m2m.compile(optimizer=opt, loss=mvae_loss)
        self.m2m.summary()
        plot_model(self.m2m, to_file='m2m_var.png')
        
        #train the neural network alternatingly
        totalEpoch = epoch
        plot_losses1 = util.PlotLosses()
        plot_losses2 = util.PlotLosses()
        history1 = History()
        history2 = History()
        
        AE_reg = np.zeros([totalEpoch, 6])
        AE_m = np.zeros([totalEpoch, 2])
        
        d2m_idxs = [22, 23, 24, 25, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
        m2m_idxs = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        
        if not load:
            for i in range(totalEpoch):
                #train main reg model
                self.d2m.fit(self.D, [self.D, self.M],        
                            epochs=1,
                            batch_size=128,
                            shuffle=True,
                            validation_split=0.2,
                            callbacks=[plot_losses1, EarlyStopping(monitor='loss', patience=60), history1])
            
                #copy loss
                AE_reg[i, :] = np.squeeze(np.asarray(list(history1.history.values())))
                                
                #copy weights for model decoder only into the second model
                for j in range(len(d2m_idxs)):
                    self.m2m.layers[m2m_idxs[j]].set_weights(self.d2m.layers[d2m_idxs[j]].get_weights())

                #train placeholder model AE
                self.m2m.fit(self.M, self.M,        
                            epochs=1,
                            batch_size=128,
                            shuffle=True,
                            validation_split=0.2,
                            callbacks=[plot_losses2, EarlyStopping(monitor='loss', patience=60), history2])

                #copy into the main model 
                for j in range(len(d2m_idxs)):
                    self.d2m.layers[d2m_idxs[j]].set_weights(self.m2m.layers[m2m_idxs[j]].get_weights())

                #copy loss
                AE_m[i, :] = np.squeeze(np.asarray(list(history2.history.values())))
 
                #write to folder for every 10th epoch for monitoring
                figs = util.plotAllLosses(AE_reg, AE_m)
                figs.savefig('Dual_Losses_var.png')
                
            #save trained model
            self.d2m.save('d2m_var.h5')
            self.m2m.save('m2m_var.h5')
        else:
            #load an already trained model
            print("Trained model loaded")
            self.d2m = load_model('d2m_var.h5')
            self.m2m = load_model('m2m_var.h5')
            
        #set all functions (copy weights from layers, incase comp. graph dont exist)
        #bug when trying to load model (related to sampling layer)
        #alternatively just set function using the active comp. graph (but model can't be saved)
         
        m_f = Input(shape=(self.mx_sz, self.my_sz, self.mz_sz))
        _ = self.m2m.layers[1](m_f)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            _ = self.m2m.layers[i](_)
        zm_f = self.m2m.layers[16](_)
        self.m2zm = Model(m_f, zm_f)
        
        zm_f_2 = Input(shape=(self.zm_dim, )) 
        _ = self.m2m.layers[17](zm_f_2)
        for i in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
            _ = self.m2m.layers[i](_)
        m_f_2 = self.m2m.layers[30](_)
        self.zm2m = Model(zm_f_2, m_f_2)
        
        d_f = Input(shape=(self.dx_sz,))
        _ = self.d2m.layers[1](d_f)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            _ = self.d2m.layers[i](_)
        zd_f = self.d2m.layers[12](_)
        self.d2zd = Model(d_f, zd_f)
        
        zd_f_2 = Input(shape=(self.zd_dim, ))
        _ = self.d2m.layers[27](zd_f_2)
        for i in [29, 31, 33, 35, 37, 39, 41]:
            _ = self.d2m.layers[i](_)
        d_f_2 = self.d2m.layers[43](_)
        self.zd2d = Model(zd_f_2, d_f_2)
        
        d_f_3 = Input(shape=(self.dx_sz,))
        _ = self.d2m.layers[1](d_f_3)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            _ = self.d2m.layers[i](_)
        zd_f_3 = self.d2m.layers[12](_)
        _ = self.d2m.layers[27](zd_f_3)
        for i in [29, 31, 33, 35, 37, 39, 41]:
            _ = self.d2m.layers[i](_)
        d_f_4 = self.d2m.layers[43](_)
        self.d2d = Model(d_f_3, d_f_4)
        
        zd_f_5 = Input(shape=(self.zd_dim, ))
        _ = self.d2m.layers[13](zd_f_5)
        for i in [14, 15, 16, 17, 18, 19, 20]:
            _ = self.d2m.layers[i](_)
        zm_f_3 = self.d2m.layers[21](_)
        self.zd2zm = Model(zd_f_5, zm_f_3)
        






def inspect_LSI(LSI, M_test, D_test, M_test_label):
    
    #check reconstructions and regression for training 
    D_train_hat = LSI.d2d.predict(LSI.D)
    M_train_hat = LSI.m2m.predict(LSI.M)
    M_train_hat_reg = LSI.d2m.predict(LSI.D)
    
    #check reconstructions and regression for testing
    D_test_hat = LSI.d2d.predict(D_test)
    M_test_hat = LSI.m2m.predict(M_test)
    M_test_hat_reg = LSI.d2m.predict(D_test)
    
    #check scatters training
    fig = plt.figure(figsize=(7, 2.5))
    plt.subplot(1, 3, 1)
    plt.scatter(LSI.D.flatten(), D_train_hat.flatten(), color='blue', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Train-Data-Recons')
    plt.subplot(1, 3, 2)
    plt.scatter(LSI.M.flatten(), M_train_hat.flatten(), color='green', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Train-Model-Recons')
    plt.subplot(1, 3, 3)
    plt.scatter(LSI.M.flatten(), M_train_hat_reg[1].flatten(), color='red', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Train-Model-Reg')
    plt.tight_layout()
    fig.savefig('readme/train_scatters.png')
    
    #check scatters testing
    fig = plt.figure(figsize=(7, 2.5))
    plt.subplot(1, 3, 1)
    plt.scatter(D_test.flatten(), D_test_hat.flatten(), color='blue', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Test-Data-Recons')
    plt.subplot(1, 3, 2)
    plt.scatter(M_test.flatten(), M_test_hat.flatten(), color='green', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Test-Model-Recons')
    plt.subplot(1, 3, 3)
    plt.scatter(M_test.flatten(), M_test_hat_reg[1].flatten(), color='red', alpha=0.1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Test-Model-Reg')
    plt.tight_layout()
    fig.savefig('readme/test_scatters.png')
    
    #histograms of reconstruction for model (binary)
    bb = np.linspace(-0.05, 1.0, 50)
    fig = plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.hist(M_test.flatten(), color='green', alpha=0.4, bins=bb)
    plt.hist(M_test_hat.flatten(), color='green', alpha=0.9, hatch='//', edgecolor='black', histtype='step', bins=bb)
    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off')
    plt.title('TestRMSE_'+str(round(RMSE(M_test, M_test_hat),4)))
    plt.subplot(1, 2, 2)
    plt.hist(LSI.M.flatten(), color='green', alpha=0.4, bins=bb)
    plt.hist(M_train_hat.flatten(), color='green', alpha=0.9, hatch='//', edgecolor='black', histtype='step', bins=bb)
    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off')
    plt.title('TrainRMSE_'+str(round(RMSE(LSI.M, M_train_hat),4)))
    plt.tight_layout()
    fig.savefig('readme/train_test_hists.png')
    
    #plot some test images with reconstructions
    num_rows = 10
    num_cols = 4
    num_images = num_rows*num_cols
    fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(M_test[i], cmap='viridis', vmin=0, vmax=1)
        if i < num_cols:
            plt.title('Reference')
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(M_test_hat[i], cmap='viridis', vmin=0, vmax=1)
        if i < num_cols:
            plt.title('Recons.')
    plt.tight_layout()
    plt.show()
    fig.savefig('readme/test_ref_recons.png')
    
    #color by label
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=9)
    cs = my_cmap(my_norm(M_test_label))

    #plot some test cases with reconstructions and regression 
    cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for case in cases:
        f = plt.figure(figsize=(12,3))
        #data and data reconstruction
        ax = f.add_subplot(1, 4, 1)
        plt.plot(D_test[case, :], ls=':', c='k', label='True', alpha=0.9)
        plt.plot(D_test_hat[case, :], c=cs[M_test_label[case]], label='Pred.', alpha=0.4)
        plt.ylim([0, 1])
        plt.title('Data ('+str(M_test_label[case])+')')
        plt.legend()

        #model (i.e. reference case)
        ax = f.add_subplot(1, 4, 2)
        plt.imshow(M_test[case, :, :, 0], cmap='viridis', vmin=0, vmax=1)
        plt.title('Ref ('+str(M_test_label[case])+')')
        plt.grid(False), plt.xticks([]), plt.yticks([])
        
        #predicted model (i.e. inversion)
        ax = f.add_subplot(1, 4, 3)
        plt.imshow(M_test_hat_reg[1][case, :, :, 0], cmap='viridis', vmin=0, vmax=1)
        plt.title('Inversion ('+str(M_test_label[case])+')')
        plt.grid(False), plt.xticks([]), plt.yticks([])
        
        #predicted model (i.e. inversion) colormap limit removed to show diffr
        ax = f.add_subplot(1, 4, 4)
        plt.imshow(M_test_hat_reg[1][case, :, :, 0], cmap='viridis')
        plt.title('Inversion ('+str(M_test_label[case])+')')
        plt.grid(False), plt.xticks([]), plt.yticks([])
        f.savefig('readme/test_sigs_ref_invs_'+str(case)+'.png')





def inspect_LSI_z(LSI, M_test, D_test, M_test_label):
    
    #get data latent variables
    zd_train = LSI.d2zd.predict(LSI.D)
    zd_test = LSI.d2zd.predict(D_test)

    #get model latent variables
    zm_train = LSI.m2zm.predict(LSI.M)
    zm_test = LSI.m2zm.predict(M_test)

    #get data to model latent variables (inversion)
    zm_train_reg = LSI.zd2zm.predict(zd_train)
    zm_test_reg = LSI.zd2zm.predict(zd_test)

    #plot distribution of zm (train) vs test
    binmax = np.max(zm_train)
    binmin = np.min(zm_train)
    bb2 = np.linspace(binmin, binmax, 50)

    fig = plt.figure(figsize=(12, 9))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.hist(zm_train[:, i].flatten(), color='green', alpha=0.4, bins=bb2, density=True)
        plt.hist(zm_test[:, i].flatten(), color='green', alpha=0.9, edgecolor='black', histtype='step', bins=bb2, density=True)
        #plt.xlim(-1.5, 1.5)
        #plt.xticks([])
        plt.grid(False), plt.yticks([])
        plt.title('$z_{m}$'+str(i+1))
    plt.tight_layout()
    fig.savefig('readme/train_test_zms.png')

    #plot distribution of zd (train) vs test
    binmax = np.max(zd_train)
    binmin = np.min(zd_train)
    bb1 = np.linspace(binmin, binmax, 50)

    fig = plt.figure(figsize=(12, 9))
    for i in range(16):
        plt.subplot(4, 5, i+1)
        plt.hist(zd_train[:, i].flatten(), color='blue', alpha=0.4, bins=bb1, density=True)
        plt.hist(zd_test[:, i].flatten(), color='blue', alpha=0.9, edgecolor='black', histtype='step', bins=bb1, density=True)
        #plt.xlim(-0.6, 0.6)
        #plt.xticks([])
        plt.grid(False), plt.yticks([])
        plt.title('$z_{d}$'+str(i+1))
    plt.tight_layout()
    fig.savefig('readme/train_test_zds.png')

    #plot distribution of zm (from recons) vs regression (from model) - training
    fig = plt.figure(figsize=(12, 9))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.hist(zm_train[:, i].flatten(), color='red', alpha=0.4, bins=bb1, density=True)
        plt.hist(zm_train_reg[:, i].flatten(), color='red', alpha=0.9, edgecolor='black', histtype='step', bins=bb1, density=True)
        #plt.xlim(-0.6, 0.6)
        #plt.xticks([])
        plt.grid(False), plt.yticks([])
        plt.title('$z_{m}$'+str(i+1))
    plt.tight_layout()
    fig.savefig('readme/train_zms_reg.png')

    #plot distribution of zm (from recons) vs regression (from model) - testing
    fig = plt.figure(figsize=(12, 9))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.hist(zm_test[:, i].flatten(), color='red', alpha=0.4, bins=bb1, density=True)
        plt.hist(zm_test_reg[:, i].flatten(), color='red', alpha=0.9, edgecolor='black', histtype='step', bins=bb1, density=True)
        #plt.xlim(-0.6, 0.6)
        #plt.xticks([])
        plt.grid(False), plt.yticks([])
        plt.title('$z_{m}$'+str(i+1))
    plt.tight_layout()
    fig.savefig('readme/test_zms_reg.png')

    #scatter plot
    fig = plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(zm_train.flatten(), zm_train_reg.flatten(), color='red', alpha=0.4)
    plt.xlabel('$z_{m}$')
    plt.ylabel('Reg-$z_{m}$')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.title('Train Reg vs Recons $z_{m}$')
    plt.subplot(1, 2, 2)
    plt.scatter(zm_test.flatten(), zm_test_reg.flatten(), color='red', alpha=0.4)
    plt.xlabel('$z_{m}$')
    plt.ylabel('Reg-$z_{m}$')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.title('Test Reg vs Recons $z_{m}$')
    plt.tight_layout()
    fig.savefig('readme/train_test_zms_scatter.png')

