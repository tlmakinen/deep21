# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.python.client import device_lib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load

## Build the Model:
def unet_baseline(dropout=0.0, n_filters=16, n_channels=32, x_dim=32):
   '''
   Baseline four-layer model, with two 2D convolutions in each layer
   '''
   ## Start with inputs
   inputs = keras.layers.Input(shape=(x_dim,x_dim,n_channels),name="image_input")
   ## First Convolutional layer made up of two convolutions, the second one down-samples
   conv1a = keras.layers.Conv2D(n_filters*1,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1b = keras.layers.Conv2D(n_filters*1,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
   conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)
   ## Second convolutional layer, essentially identical, increases the number of channels
   conv2a = keras.layers.Conv2D(n_filters*2,3,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
   #keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2b = keras.layers.Conv2D(n_filters*2,3,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
   conv2b = keras.layers.Dropout(dropout)(conv2b)
   conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)
   ## Third, continuing logically from above
   conv3a = keras.layers.Conv2D(n_filters*4,3,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
   conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
   conv3b = keras.layers.Conv2D(n_filters*4,3,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
   conv3b = keras.layers.Dropout(dropout)(conv3b)
   conv3b = keras.layers.BatchNormalization(axis=1)(conv3b)
   ## Fourth, again, continuing logically from above
   conv4a = keras.layers.Conv2D(n_filters*8,3,activation=tf.nn.relu,name="conv4a",padding="same")(conv3b)
   conv4a = keras.layers.BatchNormalization(axis=1)(conv4a)
   conv4b = keras.layers.Conv2D(n_filters*8,3,activation=tf.nn.relu,name="conv4b",padding="same",strides=2)(conv4a)
   conv4b = keras.layers.Dropout(dropout)(conv4b)
   conv4b = keras.layers.BatchNormalization(axis=1)(conv4b)
   ## middle
   #convm = keras.layers.Conv2D(512 * 2, 2, activation=tf.nn.relu, padding="same")(conv4b)
   #convm = keras.layers.Conv2D(512 * 2, 2, activation=tf.nn.relu, padding="same")(convm)
   ## symmetric upsampling path with concatenation from down-sampling 
   upconv1a = keras.layers.Conv2DTranspose(n_filters*8,3,activation=tf.nn.relu,padding="same",name="upconv1a")(conv4b)
   upconv1a = keras.layers.BatchNormalization(axis=1)(upconv1a)
   upconv1b = keras.layers.Conv2DTranspose(n_filters*8,3,activation=tf.nn.relu,padding="same",name="upconv1b",strides=2)(upconv1a)
   upconv1b = keras.layers.BatchNormalization(axis=1)(upconv1b)
   upconv1b = keras.layers.Dropout(dropout)(upconv1b) 
   ## The up-convolution is then concatenated with the output from "across the U" and passed along
   concat1 = keras.layers.concatenate([conv4a,upconv1b],name="concat1")
   concat1 = keras.layers.BatchNormalization(axis=1)(concat1)
   ## Second set of up-convolutions
   upconv2a = keras.layers.Conv2DTranspose(n_filters*4,3,activation=tf.nn.relu,padding="same",name="upconv2a")(concat1)
   upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
   upconv2b = keras.layers.Conv2DTranspose(n_filters*4,3,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
   upconv2b = keras.layers.Dropout(dropout)(upconv2b)
   concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2") 
   concat2 = keras.layers.BatchNormalization(axis=1)(concat2)
   ## Third set
   upconv3a = keras.layers.Conv2DTranspose(n_filters*2,3,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3b = keras.layers.Conv2DTranspose(n_filters*2,3,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
   upconv3b = keras.layers.Dropout(dropout)(upconv3b)
   concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
   concat3 = keras.layers.BatchNormalization(axis=1)(concat3)
   ## Fourth set, so the "U" has 4 layers 
   upconv4a = keras.layers.Conv2DTranspose(n_filters*1,3,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4b = keras.layers.Conv2DTranspose(n_filters*1,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
   #upconv4b = keras.layers.Dropout(dropout)(upconv4b)
   concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
   concat4 = keras.layers.BatchNormalization(axis=1)(concat4)
   ## Output is then put in to a shape to match the original data
   output = keras.layers.Conv2DTranspose(n_channels,1,padding="same",name="output")(concat4)
   ## Compile the model
   model = keras.models.Model(inputs=inputs,outputs=output)
   return model


def unet_3Layer3conv(dropout=0.0, n_filters=16, n_channels=32, x_dim=32):
   '''
   simpler model, three layers deep
   three convolutions per layer
   '''
   ## Start with inputs
   inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
## First Convolutional layer made up of two convolutions, the second one down-samples
   conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a1",padding="same")(conv1a)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1b = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
   conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)    
   ## Second convolutional layer, essentially identical, increases the number of channels
   conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
   conv2a = keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a1",padding="same")(conv2a)
   conv2a = keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2b = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
   conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)    
   ## Third, continuing logically from above
   conv3a = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
   conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
   conv3a = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3a1",padding="same")(conv3a)
   conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
   conv3b = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
   conv3b = keras.layers.BatchNormalization(axis=1)(conv3b)
   ## First set of up-convolutions
   upconv2a = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2a")(conv3b)
   upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
   upconv2a = keras.layers.Conv2D(256,3,activation=tf.nn.relu,padding="same",name="upconv2a1")(upconv2a)
   upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
   upconv2b = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
   concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2") 
   concat2 = keras.layers.BatchNormalization(axis=1)(concat2)
   ## Second set
   upconv3a = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,padding="same",name="upconv3a1")(upconv3a)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3b = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
   concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
   concat3 = keras.layers.BatchNormalization(axis=1)(concat3)
   ## Third set, so the "U" has 3 layers 
   upconv4a = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,padding="same",name="upconv4a1")(upconv4a)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4b = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
   concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
   ## Output is then put in to a shape to match the original data
   output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)
   ## Compile the model
   model = keras.models.Model(inputs=inputs,outputs=output)

   return model

def unet_2Layer3conv(dropout=0.0):
   '''
   simpler model, two layers deep,
   three convolutions per layer
   '''
   ## Start with inputs
   inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
## First Convolutional layer made up of two convolutions, the second one down-samples
   conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a1",padding="same")(conv1a)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1b = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
   conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)    
   ## Second convolutional layer, essentially identical, increases the number of channels
   conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
   conv2a = keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a1",padding="same")(conv2a)
   conv2a = keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2b = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
   conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)    

   upconv3a = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3a")(conv2b)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,padding="same",name="upconv3a1")(upconv3a)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3b = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
   concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
   concat3 = keras.layers.BatchNormalization(axis=1)(concat3)
   ## Third set, so the "U" has 3 layers 
   upconv4a = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,padding="same",name="upconv4a1")(upconv4a)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4b = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
   concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
   ## Output is then put in to a shape to match the original data
   output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)
   ## Compile the model
   model = keras.models.Model(inputs=inputs,outputs=output)

   return model

def unet_3Layer2conv(dropout=0.0):
   '''
   3 layers deep,
   2 convolutions per layer
   '''
   ## Start with inputs
   inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
   ## First Convolutional layer made up of two convolutions, the second one down-samples
   conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
   conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
   conv1b = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
   conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)
   ## Second convolutional layer, essentially identical, increases the number of channels
   conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
   #keras.layers.BatchNormalization(axis=1)(conv1b)
   conv2b = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
   conv2b = keras.layers.Dropout(dropout)(conv2b)
   conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)
   ## Third, continuing logically from above
   conv3a = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
   conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
   conv3b = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
   conv3b = keras.layers.Dropout(dropout)(conv3b)
   conv3b = keras.layers.BatchNormalization(axis=1)(conv3b)
   ## First set of up-convolutions
   upconv2a = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2a")(conv3b)
   upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
   upconv2b = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
   upconv2b = keras.layers.Dropout(dropout)(upconv2b)
   concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2") 
   concat2 = keras.layers.BatchNormalization(axis=1)(concat2)
   ## Second set
   upconv3a = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
   upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
   upconv3b = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
   upconv3b = keras.layers.Dropout(dropout)(upconv3b)
   concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
   concat3 = keras.layers.BatchNormalization(axis=1)(concat3)
   ## Third set, so the "U" has 4 layers 
   upconv4a = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
   upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
   upconv4b = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
   #upconv4b = keras.layers.Dropout(dropout)(upconv4b)
   concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
   ## Output is then put in to a shape to match the original data
   output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)
   ## Compile the model
   model = keras.models.Model(inputs=inputs,outputs=output)
   return model

def unet_5layer2conv(dropout=0.0):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    conv1a = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
    conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
    conv1b = keras.layers.Conv2D(64,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
    conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)
    ## Second convolutional layer, essentially identical, increases the number of channels
    conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
    #keras.layers.BatchNormalization(axis=1)(conv1b)
    conv2b = keras.layers.Conv2D(128,3,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
    conv2b = keras.layers.Dropout(dropout)(conv2b)
    conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)
    ## Third, continuing logically from above
    conv3a = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
    conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
    conv3b = keras.layers.Conv2D(256,3,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
    conv3b = keras.layers.Dropout(dropout)(conv3b)
    conv3b = keras.layers.BatchNormalization(axis=1)(conv3b)
    ## Fourth, again, continuing logically from above
    conv4a = keras.layers.Conv2D(512,3,activation=tf.nn.relu,name="conv4a",padding="same")(conv3b)
    conv4a = keras.layers.BatchNormalization(axis=1)(conv4a)
    conv4b = keras.layers.Conv2D(512,3,activation=tf.nn.relu,name="conv4b",padding="same",strides=2)(conv4a)
    conv4b = keras.layers.Dropout(dropout)(conv4b)
    conv4b = keras.layers.BatchNormalization(axis=1)(conv4b)
    ## middle
    convm = keras.layers.Conv2D(512 * 2, 3, activation=tf.nn.relu, padding="same")(conv4b)
    convm = keras.layers.BatchNormalization(axis=1)(convm)
    convm = keras.layers.Conv2D(512 * 2, 3, activation=tf.nn.relu, padding="same")(convm)
    convm = keras.layers.BatchNormalization(axis=1)(convm)
    ## symmetric upsampling path with concatenation from down-sampling 
    upconv1a = keras.layers.Conv2DTranspose(512,3,activation=tf.nn.relu,padding="same",name="upconv1a")(convm)
    upconv1a = keras.layers.BatchNormalization(axis=1)(upconv1a)
    upconv1b = keras.layers.Conv2DTranspose(512,3,activation=tf.nn.relu,padding="same",name="upconv1b",strides=2)(upconv1a)
    upconv1b = keras.layers.BatchNormalization(axis=1)(upconv1b)
    upconv1b = keras.layers.Dropout(dropout)(upconv1b) 
    ## The up-convolution is then concatenated with the output from "across the U" and passed along
    concat1 = keras.layers.concatenate([conv4a,upconv1b],name="concat1")
    concat1 = keras.layers.BatchNormalization(axis=1)(concat1)
    ## Second set of up-convolutions
    upconv2a = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2a")(concat1)
    upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
    upconv2b = keras.layers.Conv2DTranspose(256,3,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
    upconv2b = keras.layers.Dropout(dropout)(upconv2b)
    concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2") 
    concat2 = keras.layers.BatchNormalization(axis=1)(concat2)
    ## Third set
    upconv3a = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
    upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
    upconv3b = keras.layers.Conv2DTranspose(128,3,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
    upconv3b = keras.layers.Dropout(dropout)(upconv3b)
    concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
    concat3 = keras.layers.BatchNormalization(axis=1)(concat3)

    ## Fourth set, so the "U" has 4 layers 
    upconv4a = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
    upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
    upconv4b = keras.layers.Conv2DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
    upconv4b = keras.layers.Dropout(dropout)(upconv4b)
    concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
    #concat4 = keras.layers.BatchNormalization(axis=1)(concat4)
    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model


def unet_3D(dropout=0.0):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(64,64,30,1),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    conv1a = keras.layers.Conv3D(64,3,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
    conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
    conv1b = keras.layers.Conv3D(64,3,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
    conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)

    ## Fourth set, so the "U" has 4 layers 
    upconv4a = keras.layers.Conv3DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4a")(conv1b)
    upconv4a = keras.layers.BatchNormalization(axis=1)(upconv4a)
    upconv4b = keras.layers.Conv3DTranspose(64,3,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
    upconv4b = keras.layers.BatchNormalization(axis=1)(upconv4b)
    #upconv4b = keras.layers.Dropout(dropout)(upconv4b)
    concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
    #concat4 = keras.layers.BatchNormalization(axis=1)(concat4)

    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv3DTranspose(1,1,padding="same",name="output")(concat4)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model