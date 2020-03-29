import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate, MaxPool2D, Activation

def conv2d_nblock(input_tensor, n_filters, n_layers, kernel_size = 3, activation='relu', \
                                batchnorm = True, momentum=0.1, strides=1):
    """Function to add n_blocks convolutional layers with the parameters passed to it"""
    
    x = input_tensor
    
    for n in range(n_layers):        
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
              padding = 'same', strides=strides)(x)
             
        x = Activation(activation)(x)        
        if batchnorm:
            x = BatchNormalization(momentum=momentum)(x)   
        return x
    
    
def conv2d_single_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, 
                                            strides=2, momentum=0.99, activation='relu', maxpool=False, dropout=0.0):
    """Function to add 1 convolutional layer with the parameters passed to it 
                                            (downsample optional; determined by strides)"""
    # first layer
    if maxpool:
        x = MaxPool3D(pool_size=(strides,strides,strides), padding='same')(input_tensor)
    
    else:
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size, kernel_size),\
                          padding = 'same', strides=strides)(input_tensor)
        
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Activation(activation)(x)
    
    if batchnorm:
        x = BatchNormalization(momentum=momentum, epsilon=1e-5)(x)
    
    return x

def unet_4conv(n_filters = 16, n_channels = 32, x_dim=32, dropout=0.0):
    '''
    Baseline unet model for inputs of shape (x_dim,x_dim,n_channels)
    '''

    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, n_channels),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv2d_nblock(inputs, n_filters*1, batchnorm = True)
    x = conv2d_single_block(x1, n_filters*1, batchnorm=True, strides=2)
    
    
    x2 = conv2d_nblock(x, n_filters*2, batchnorm=True)
    x = conv2d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv2d_nblock(x, n_filters*4, batchnorm=True)
    x = conv2d_single_block(x3, n_filters*4, batchnorm=True, strides=2)
    
    x4 = conv2d_nblock(x, n_filters*8, batchnorm=True)
    x = conv2d_single_block(x4, n_filters*8, batchnorm=True, strides=2)

    x5 = conv2d_nblock(x, n_filters*16, batchnorm=True, strides=1)
    
   
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
    x1 = conv2d_nblock(inputs, n_filters*1, batchnorm = True)
    x = conv2d_single_block(x1, n_filters*1, batchnorm=True, strides=2)


    x2 = conv2d_nblock(x, n_filters*2, batchnorm=True)
    x = conv2d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv2d_nblock(x, n_filters*4, batchnorm=True)
    x = conv2d_single_block(x3, n_filters*4, batchnorm=True, strides=2)

    x4 = conv2d_nblock(x, n_filters*8, batchnorm=True)

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

def unet_5conv(n_filters = 16, n_channels = 32, x_dim=32, dropout=0.0):
    '''
    Baseline unet model for inputs of shape (x_dim,x_dim,n_channels)
    '''

    ## Start with inputs
    inputs = keras.layers.Input(shape=(x_dim, x_dim, n_channels),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    x1 = conv2d_nblock(inputs, n_filters*1, batchnorm = True)
    x = conv2d_single_block(x1, n_filters*1, batchnorm=True, strides=2)
    
    
    x2 = conv2d_nblock(x, n_filters*2, batchnorm=True)
    x = conv2d_single_block(x2, n_filters*2, batchnorm=True, strides=2)

    x3 = conv2d_nblock(x, n_filters*4, batchnorm=True)
    x = conv2d_single_block(x3, n_filters*4, batchnorm=True, strides=2)
    
    x4 = conv2d_nblock(x, n_filters*8, batchnorm=True)
    x = conv2d_single_block(x4, n_filters*8, batchnorm=True, strides=2)

    x5 = conv2d_nblock(x, n_filters*16, batchnorm=True)
    x = conv2d_single_block(x5, n_filters*16, batchnorm=True, strides=2)

    # bottom
    x = conv2d_nblock(x, n_filters*32, batchnorm=True)

    # expansive path 
    xl = Conv2DTranspose(n_filters*16, kernel_size=3, strides=2, padding='same')(x)    
    x = concatenate([xl, x5])
    x = conv2d_single_block(x, n_filters*16, kernel_size=3, strides=1, momentum=0.99)
      
    x6 = Conv2DTranspose(n_filters*8, kernel_size=3, strides=2, padding='same')(x)    
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
