## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose, concatenate, MaxPool3D, Activation

def conv3d_double_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=1):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x1 = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same')(input_tensor)

    if batchnorm:
        x = BatchNormalization()(x1)    
    x = Activation('relu')(x1)
    
    # second layer downsamples
    x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same', strides=strides)(input_tensor)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def conv3d_single_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=2, momentum=0.99):
    """Function to add 1 convolutional layer with the parameters passed to it"""
    # first layer
    x1 = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same', strides=strides)(input_tensor)

    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x1)
    x = Activation('relu')(x1)

    return x

def build_unet3d(n_filters = 16, n_cubes=1, x_dim=32, dropout = 0.0, batchnorm = True):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim,n_cubes),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    #n_filters = n_cubes
    x1 = conv3d_double_block(inputs, n_filters*1, batchnorm = True)
    x = conv3d_single_block(x1, n_filters*1, batchnorm=True, strides=2)

    x2 = conv3d_double_block(x, n_filters  * 2, batchnorm=True)
    x = conv3d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv3d_double_block(x, n_filters*4, batchnorm=True)
    x = conv3d_single_block(x3, n_filters*4, batchnorm=True, strides=2)

    x4 = conv3d_double_block(x, n_filters*8, batchnorm=True)
    x = conv3d_single_block(x4, n_filters*8, batchnorm=True, strides=2)

    x5 = conv3d_double_block(x, n_filters*16, batchnorm=True, strides=1)

    # expansive path    
    x6 = Conv3DTranspose(n_filters*8, kernel_size=3, strides=(2,2,2), padding='same')(x5)    
    x = concatenate([x6, x4])
    x = conv3d_single_block(x, n_filters*8, kernel_size=3, strides=1, momentum=0.1)

    x7 = Conv3DTranspose(n_filters*4, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x7, x3])
    x = conv3d_single_block(x, n_filters*4, kernel_size=3, strides=1)

    x8 = Conv3DTranspose(n_filters*2, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x8,x2])
    x = conv3d_single_block(x, n_filters*2, kernel_size=3, strides=1, momentum=0.1)

    x9 = Conv3DTranspose(n_filters*1, kernel_size=3, strides=(2,2,2), padding='same')(x)
    x = concatenate([x9, x1])
    x = conv3d_single_block(x, n_filters*1, kernel_size=3, strides=1, momentum=0.1)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv3DTranspose(n_cubes,1,padding="same",name="output")(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model



def build_unet3d_maxpool(n_filters = 32, n_cubes=1, dropout = 0.0, batchnorm = True):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(n_filters, n_filters, n_filters,n_cubes),name="image_input")
    # contractive path
    x1,x = conv3d_double_block(inputs, n_filters*1, batchnorm = True, strides=1)
    x = MaxPool3D((2,2,2))(x)
    
    x2,x = conv3d_double_block(x, n_filters*2, batchnorm=True, strides=1)
    x = MaxPool3D((2,2,2))(x)
    
    x3,x = conv3d_double_block(x, n_filters*4, batchnorm=True, strides=1)
    x = MaxPool3D((2,2,2))(x)
    
    x4,x = conv3d_double_block(x, n_filters*8, batchnorm=True, strides=1)
    x = MaxPool3D((2,2,2))(x)
    
    _,x = conv3d_double_block(x, n_filters*16, batchnorm=True, strides=1)
    
    # expansive path    
    x6 = Conv3DTranspose(n_filters*8, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x6, x4])
    _,x = conv3d_double_block(x, n_filters*8, kernel_size=3, strides=1)
    
    x7 = Conv3DTranspose(n_filters*4, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x7, x3])
    _,x = conv3d_double_block(x, n_filters*4, kernel_size=3, strides=1)
    
    x8 = Conv3DTranspose(n_filters*2, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x8,x2])
    _,x = conv3d_double_block(x, n_filters*2, kernel_size=3, strides=1)
    
    x9 = Conv3DTranspose(n_filters*1, kernel_size=3, strides=(2,2,2), padding='same')(x)
    x = concatenate([x9, x1])
    _,x = conv3d_double_block(x, n_filters*1, kernel_size=3, strides=1, batchnorm=False)
    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv3DTranspose(n_cubes,1,padding="same",name="output")(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model

def build_unet3d_3conv(n_filters = 32, n_cubes=1, x_dim = 32, dropout = 0.0, batchnorm = True):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim,n_cubes),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv3d_double_block(inputs, n_filters*1, batchnorm = True)
    x = conv3d_single_block(x1, n_filters*1, batchnorm=True, strides=2)

    x2 = conv3d_double_block(x, n_filters  * 2, batchnorm=True)
    x = conv3d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv3d_double_block(x, n_filters*4, batchnorm=True)
    x = conv3d_single_block(x3, n_filters*4, batchnorm=True, strides=2)

    x4 = conv3d_double_block(x, n_filters*8, batchnorm=True, strides=1)

    # expansive path
    x7 = Conv3DTranspose(n_filters*4, kernel_size=3, strides=(2,2,2), padding='same')(x4)    
    x = concatenate([x7, x3])
    x = conv3d_single_block(x, n_filters*4, kernel_size=3, strides=1, momentum=0.1)

    x8 = Conv3DTranspose(n_filters*2, kernel_size=3, strides=(2,2,2), padding='same')(x)    
    x = concatenate([x8,x2])
    x = conv3d_single_block(x, n_filters*2, kernel_size=3, strides=1, momentum=0.1)

    x9 = Conv3DTranspose(n_filters*1, kernel_size=3, strides=(2,2,2), padding='same')(x)
    x = concatenate([x9, x1])
    x = conv3d_single_block(x, n_filters*1, kernel_size=3, strides=1, momentum=0.1)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv3DTranspose(n_cubes,1,padding="same",name="output")(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model