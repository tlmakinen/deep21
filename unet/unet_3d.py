## 3D UNet for 21cm Observation De-Noising
## by TLM
## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose, concatenate,\
                                    MaxPool3D, Activation, LeakyReLU, Dropout

class unet3D():  
    """
    General class for building fully connected 3D convolutional UNet
        Parameters: `n_filters`: starting filter size
                    `conv_width`: how many convolutions to be performed in residual block
                    `network_depth`: how many layers deep your network goes 
                        (limit: growth_factor^network_depth =< x_dim)
                    `growth_factor`: 2 (how to divide feature size)
                    `n_cubes_in`: how many image cubes to put into 4D inputs
                    `n_cubes_out`: number of cubes you want out
                    `x_dim`: image input size (x_dim, x_dim, x_dim)
                    `batchnorm`: bool (usually True) to reduce internal covariance shift
                    `momentum`: batchnorm param, set to 0.1 for outputs, (default=0.9)
                    `epsilon`: batchnorm param (default = 1e-5) 
                    `activation`: activation function for outputs (default = 'relu')
                    `maxpool`: whether or not to use MaxPool feature to downsample (default = True)
    """

    def __init__(self, n_filters = 16, conv_width=1, 
                 network_depth = 4,
                 n_cubes_in=1, n_cubes_out=1,
                 x_dim=32, dropout = 0.0, 
                 growth_factor=2, batchnorm_in=True,
                 batchnorm_down=True, batchnorm_up=False,
                 batchnorm_out=False,
                 out_act = False, 
                 momentum=0.1, epsilon=1e-5,
                 activation='relu', maxpool=False
                 ):
        
        self.n_filters = n_filters
        self.n_cubes_in = n_cubes_in
        self.n_cubes_out = n_cubes_out
        self.conv_width = conv_width
        self.network_depth = network_depth
        self.x_dim = x_dim
        self.dropout = dropout
        self.growth_factor = growth_factor
        self.batchnorm_in = batchnorm_in
        self.batchnorm_down = batchnorm_down
        self.batchnorm_up = batchnorm_up
        self.batchnorm_out = batchnorm_out
        self.out_act = out_act,
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.maxpool = maxpool        
        
        # define all layers
        
    def conv_block(self, input_tensor, n_filters, n_layers=1, strides=1, kernel_size=3, \
                           momentum=0.9, maxpool=False, batchnorm=True, layer_num=None):
        """Function to add n_blocks convolutional layers with the parameters passed to it"""
        if layer_num is not None:
            if strides > 1:
                name = 'downsample_{}'.format(layer_num)
        else:
            name = None
        
        x = input_tensor    
        
        if maxpool:
            x = MaxPool3D(pool_size=(strides,strides), padding='same')(x)
            if batchnorm:
                x = BatchNormalization(momentum=momentum)(x)   
            x = Activation(self.activation)(x)   
            
        
        else:
            for l in range(n_layers):        
                x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
                      padding = 'same', strides=strides, name=name)(x)

                if batchnorm:
                    x = BatchNormalization(momentum=momentum)(x)   
                x = Activation(self.activation)(x)    
            return x           
                     
    
    def build_model(self):
        network_depth = self.network_depth
        n_filters = self.n_filters
        growth_factor = self.growth_factor
        x_dim = self.x_dim
        momentum = self.momentum

        # start with inputs
        inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim, self.n_cubes_in),name="image_input")
        x = inputs
        concat_down = []
        # downsample path
        for l in range(network_depth):
            x = self.conv_block(x, n_filters, n_layers=self.conv_width,strides=1, batchnorm=self.batchnorm_in) 
            concat_down.append(x)
            x = self.conv_block(x, n_filters, n_layers=1, batchnorm=self.batchnorm_down, strides=2, 
                                    maxpool=self.maxpool, layer_num=l+1)
            n_filters *= growth_factor
        
        # reverse order of down layers
        concat_down = concat_down[::-1]  
        # middle
        x = self.conv_block(x, n_filters, n_layers=self.conv_width, strides=1)
        
        # expansive path
        n_filters //= growth_factor
        for l in range(network_depth):
            x = Conv3DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            if self.batchnorm_up:            
                x = BatchNormalization(momentum=momentum, epsilon=self.epsilon)(x)
            if self.out_act:
                x = Activation(self.activation)(x)
            x = concatenate([x, concat_down[l]])
            x = self.conv_block(x, n_filters, n_layers=self.conv_width, kernel_size=3, 
                                        strides=1, batchnorm=self.batchnorm_out, momentum=self.momentum)   
            n_filters //= growth_factor
            
        output = Conv3DTranspose(self.n_cubes_out,1,padding="same",name="output")(x)

        # return model
        model = keras.models.Model(inputs=inputs,outputs=output)
        return model
