import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate, MaxPool2D, Activation

def conv2d_double_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=1):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x1 = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x1)
    x = Activation('relu')(x1)
    
    # second layer downsamples
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              padding = 'same', strides=strides)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
        
    return x

def conv2d_single_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides=2, momentum=0.99):
    """Function to add 1 downsampling convolutional layer with the parameters passed to it"""
    # second layer downsamples
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              padding = 'same', strides=strides)(input_tensor)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    x = Activation('relu')(x)
        
    return x


def unet_4conv(n_filters = 16, n_channels = 32, x_dim=32, dropout=0.0):
    '''
    Baseline unet model for inputs of shape (x_dim,x_dim,n_channels)
    '''

    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, n_channels),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv2d_double_block(inputs, n_filters*1, batchnorm = True)
    x = conv2d_single_block(x1, n_filters*1, batchnorm=True, strides=2)
    
    
    x2 = conv2d_double_block(x, n_filters*2, batchnorm=True)
    x = conv2d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv2d_double_block(x, n_filters*4, batchnorm=True)
    x = conv2d_single_block(x3, n_filters*4, batchnorm=True, strides=2)
    
    x4 = conv2d_double_block(x, n_filters*8, batchnorm=True)
    x = conv2d_single_block(x4, n_filters*8, batchnorm=True, strides=2)

    x5 = conv2d_double_block(x, n_filters*16, batchnorm=True, strides=1)
    
   
    # expansive path    
    x6 = Conv2DTranspose(n_filters*8, kernel_size=3, strides=2, padding='same')(x5)    
    x = concatenate([x6, x4])
    x = conv2d_single_block(x, n_filters*8, kernel_size=3, strides=1, momentum=0.99)
    
    x7 = Conv2DTranspose(n_filters*4, kernel_size=3, strides=2, padding='same')(x)    
    x = concatenate([x7, x3])
    x = conv2d_single_block(x, n_filters*4, kernel_size=3, strides=1, momentum=0.99)
    
    x8 = Conv2DTranspose(n_filters*2, kernel_size=3, strides=2, padding='same')(x)    
    x = concatenate([x8,x2])
    x = conv2d_single_block(x, n_filters*2, kernel_size=3, strides=1, momentum=0.99)
    
    x9 = Conv2DTranspose(n_filters*1, kernel_size=3, strides=2, padding='same')(x)
    x = concatenate([x9, x1])
    x = conv2d_single_block(x, n_filters*1, kernel_size=3, strides=1, momentum=0.99)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv2DTranspose(n_channels,1,padding="same",name="output")(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    
    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model

def unet_3conv(n_filters = 16, n_channels=32, x_dim = 32, dropout=0.0):
    '''
    shallower model for testing    
    '''

    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, n_channels),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv2d_double_block(inputs, n_filters*1, batchnorm = True)
    x = conv2d_single_block(x1, n_filters*1, batchnorm=True, strides=2)


    x2 = conv2d_double_block(x, n_filters*2, batchnorm=True)
    x = conv2d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv2d_double_block(x, n_filters*4, batchnorm=True)
    x = conv2d_single_block(x3, n_filters*4, batchnorm=True, strides=2)

    x4 = conv2d_double_block(x, n_filters*8, batchnorm=True)

    # expansive path
    x7 = Conv2DTranspose(n_filters*4, kernel_size=3, strides=2, padding='same')(x4)    
    x = concatenate([x7, x3])
    x = conv2d_single_block(x, n_filters*4, kernel_size=3, strides=1, momentum=0.99)

    x8 = Conv2DTranspose(n_filters*2, kernel_size=3, strides=2, padding='same')(x)    
    x = concatenate([x8,x2])
    x = conv2d_single_block(x, n_filters*2, kernel_size=3, strides=1, momentum=0.99)

    x9 = Conv2DTranspose(n_filters*1, kernel_size=3, strides=2, padding='same')(x)
    x = concatenate([x9, x1])
    x = conv2d_single_block(x, n_filters*1, kernel_size=3, strides=1, momentum=0.99)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv2DTranspose(n_channels,1,padding="same",name="output")(x)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model