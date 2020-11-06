## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose, concatenate,\
                                    MaxPool3D, Activation, LeakyReLU, Dropout


def conv3d_double_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=1, activation='relu', dropout=True, momentum=0.1):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x1 = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same')(input_tensor)
    if batchnorm:
        x1 = BatchNormalization(momentum=momentum, epsilon=1e-5)(x1)
    x1 = Activation(activation)(x1)
    
    # second layer downsamples
    x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same', strides=strides)(x1)
    if batchnorm:
        x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    x = Activation(activation)(x)
        
    return x1,x

def conv3d_single_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, 
                                            strides=2, momentum=0.99, activation='relu', maxpool=False, dropout=0.0):
    """Function to add 1 convolutional layer with the parameters passed to it 
                                            (downsample optional; determined by strides)"""
    # first layer
    if maxpool:
        x1 = MaxPool3D(pool_size=(strides,strides,strides), padding='same')(input_tensor)
    
    else:
        x1 = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
                          padding = 'same', strides=strides)(input_tensor)
    if batchnorm:
        x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x1)
        
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Activation(activation)(x1)
    
    return x

def unet(n_filters = 16, n_cubes_in=2, n_cubes_out=1, x_dim=32, dropout = 0.0, 
                            growth_factor=2, batchnorm = True, momentum=0.1, activation='relu', maxpool=False):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim,n_cubes_in),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x11,x12 = conv3d_double_block(inputs, n_filters*1, batchnorm = True, activation='relu')       # layer 1
    x = conv3d_single_block(x12, n_filters*1, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)
    
    n_filters *= growth_factor
    x21,x22 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 2
    x = conv3d_single_block(x22, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)
    
    n_filters *= growth_factor
    x31,x32 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 3
    x = conv3d_single_block(x32, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)
    
    n_filters *= growth_factor
    x41,x42 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 4
    x = conv3d_single_block(x42, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)
    
    n_filters *= growth_factor
    x51,x52 = conv3d_double_block(x, n_filters, batchnorm=True, strides=1, activation = activation)  # layer 5 (middle)
    #x = conv3d_single_block(x5, n_filters*16, batchnorm=True, strides=1)
    
    n_filters //= growth_factor
    
    # expansive path    
    
    x6 = conv3d_single_block(x52, n_filters, kernel_size=3, strides=1, momentum=momentum, activation=activation)   
    x = concatenate([x6, x52])
    x6 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum, activation=activation)   
    x = concatenate([x6, x51])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 6
    x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    x = Activation(activation)(x)
#     x = Conv3DTranspose(n_filters*8, kernel_size=3, strides=1, padding='same', activation='relu')(x) 
#     x = BatchNormalization()(x)
  
    
    n_filters //= growth_factor
    x7 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x7, x42])
    x7 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x7, x41])    
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 7
    x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    x = Activation(activation)(x)

    
    n_filters //= growth_factor
    x8 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x8,x32])
    x8 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x8,x31])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 8
    x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    x = Activation(activation)(x)

    
    n_filters //= growth_factor
    x9 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    x = concatenate([x9, x22])
    x9 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    x = concatenate([x9, x21])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 9
    x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    x = Activation(activation)(x)

    
    x10 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    x = concatenate([x10, x12])
    x10 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    x = concatenate([x10, x11])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    output = Conv3DTranspose(n_cubes_out,1,padding="same",name="output", activation='selu')(x)              # layer 10
    
    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model