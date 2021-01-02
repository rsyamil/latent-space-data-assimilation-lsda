import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

import keras
from IPython.display import clear_output
import matplotlib as mpl

#plot function for sample images
def plot_tile(samples):
    
    num_samples, x_dim, y_dim, _ = samples.shape
    axes = (np.round(np.sqrt(num_samples))).astype(int)
    fig = plt.figure(figsize=(axes, axes))
    gs = gridspec.GridSpec(axes, axes)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(sample, cmap=plt.get_cmap('viridis'), aspect='auto')
        
        
#visualize the generated signals (for training dataset) 
def plot_signals(y_reg_train, labels):

    fig, ax = plt.subplots(1,1, figsize = (16, 7))

    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=9)
    cs = my_cmap(my_norm(labels))

    for j in range(10):
        plt.subplot(2, 5, j+1)
        for i in range(500):
            if (labels[i] == j):
                plt.plot(y_reg_train[i, :], c=cs[i], alpha=0.5)
        plt.ylim([0, 1])
        plt.title('digit '+str(j))
        
    return fig

#function to view training and validation losses
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss", c = 'green')
        plt.plot(self.x, self.val_losses, label="val_loss", c = 'red')
        plt.legend()
        plt.show()
        
#function to view multiple losses
def plotAllLosses(loss1, loss2):         
    N, m1f = loss1.shape
    _, m2f = loss2.shape
    
    print(loss1.shape)
    print(loss2.shape)
    
    fig = plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.plot(loss1[:, 0], label='loss1_check1')
    plt.plot(loss1[:, 1], label='loss1_check2')
    plt.plot(loss1[:, 2], label='loss1_check3')
    plt.plot(loss1[:, 3], label='loss1_check4')
    plt.plot(loss1[:, 4], label='loss1_check3')
    plt.plot(loss1[:, 5], label='loss1_check4')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(loss2[:, 0], label='loss2_check1')
    plt.plot(loss2[:, 1], label='loss2_check2')
    plt.legend()
    
    return fig
    