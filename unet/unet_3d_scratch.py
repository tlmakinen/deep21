## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose, concatenate,\
                                    MaxPool3D, Activation, LeakyReLU, Dropout

def conv3d_double_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=1, activation='relu', dropout=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x1 = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x1)
    x = Activation(activation)(x1)
    
    # second layer downsamples
    x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same', strides=strides)(input_tensor)

    x = Activation(activation)(x)

    if batchnorm:
        x = BatchNormalization()(x)
        
    return x

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
        
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Activation(activation)(x1)

    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x1)
    
    return x

def build_unet3d(n_filters = 16, n_cubes=1, x_dim=32, dropout = 0.0, activation='relu', batchnorm = True):
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

def unet3d(n_filters = 16, n_cubes_in=2, n_cubes_out=1, x_dim=32, dropout = 0.0, activation='relu',
                            growth_factor=2, batchnorm = True, momentum=0.1):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim,n_cubes_in),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv3d_double_block(inputs, n_filters*1, batchnorm = True)       # layer 1
    x = conv3d_single_block(x1, n_filters*1, batchnorm=True, strides=2)

    n_filters *= growth_factor
    x2 = conv3d_double_block(x, n_filters, batchnorm=True)              # layer 2
    x = conv3d_single_block(x2, n_filters, batchnorm=True, strides=2)

    n_filters *= growth_factor
    x3 = conv3d_double_block(x, n_filters, batchnorm=True)              # layer 3
    x = conv3d_single_block(x3, n_filters, batchnorm=True, strides=2)

    n_filters *= growth_factor
    x4 = conv3d_double_block(x, n_filters, batchnorm=True)              # layer 4
    x = conv3d_single_block(x4, n_filters, batchnorm=True, strides=2)

    n_filters *= growth_factor
    x5 = conv3d_double_block(x, n_filters, batchnorm=True, strides=1)  # layer 5


    n_filters //= growth_factor
    # expansive path    
    x6 = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x5)    # layer 6
    x = concatenate([x6, x4])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum)     

    n_filters //= growth_factor
    x7 = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 7
    x = concatenate([x7, x3])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum)

    n_filters //= growth_factor
    x8 = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 8
    x = concatenate([x8,x2])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum)

    n_filters //= growth_factor
    x9 = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 9
    x = concatenate([x9, x1])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv3DTranspose(n_cubes_out,1,padding="same",name="output", activation='selu')(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model

def unet3d_con(n_filters = 16, n_cubes_in=2, n_cubes_out=1, x_dim=32, dropout = 0.0, 
                            growth_factor=2, batchnorm = True, momentum=0.1, activation='relu', maxpool=False):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, x_dim,n_cubes_in),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv3d_double_block(inputs, n_filters*1, batchnorm = True, activation=activation)       # layer 1
    x = conv3d_single_block(x1, n_filters*1, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)

    n_filters *= growth_factor
    x2 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 2
    x = conv3d_single_block(x2, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)

    n_filters *= growth_factor
    x3 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 3
    x = conv3d_single_block(x3, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)

    n_filters *= growth_factor
    x4 = conv3d_double_block(x, n_filters, batchnorm=True, activation=activation)              # layer 4
    x = conv3d_single_block(x4, n_filters, batchnorm=True, strides=2, activation=activation, maxpool=maxpool)

    n_filters *= growth_factor
    x5 = conv3d_double_block(x, n_filters, batchnorm=True, strides=1, activation = activation)  # layer 5 (middle)


    # expansive path    
    n_filters //= growth_factor
    x6 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, momentum=momentum, activation=activation)   
    x = concatenate([x6, x5])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 6
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    n_filters //= growth_factor
    x7 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1, batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x7, x4])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 7
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    n_filters //= growth_factor
    x8 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=batchnorm, momentum=momentum, activation=activation)
    x = concatenate([x8,x3])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 8
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    n_filters //= growth_factor
    x9 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=False, momentum=momentum, activation=activation)
    x = concatenate([x9, x2])
    x = Conv3DTranspose(n_filters, kernel_size=3, strides=(2,2,2), padding='same')(x)    # layer 9
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    x10 = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    x = concatenate([x10, x1])
    x = conv3d_single_block(x, n_filters, kernel_size=3, strides=1,  batchnorm=True, momentum=momentum, activation=activation)
    output = Conv3DTranspose(n_cubes_out,1,padding="same",name="output", activation='selu')(x)              # layer 10


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