# optimize hyperparameters for 3D UNet
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tensorflow.python.client import device_lib
import pickle
from sklearn.externals.joblib import dump, load
import h5py
import healpy as hp
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import multiprocessing
# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_3d
from data_utils import dataloaders

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]

def build_compile(net, params, N_GPU):

    model = net.build_model()
    if N_GPU > 1:
        model = keras.utils.multi_gpu_model(model, gpus=N_GPU)
 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=True), loss="mse",metrics=["mse"])
    return model

# DEFINE INPUT PARAMS
params = {
    'n_filters' : 16,
    'n_cubes_in': 1,
    'n_cubes_out': 1,
    'conv_width' : 1,
    'network_depth': 3,
    'batch_size' : 48,
    'num_epochs' : 200,
    'act' : 'relu',
    'lr': 0.0001,
    'out_dir': 'trials/',
    'nu_indx': np.arange(32),
    'load_weights': False,
    }

def train_unet(params):
 
    # initialize model
    model = unet_3d.unet3D(n_filters=16, conv_width=params['conv_width'],
                        network_depth=3, n_cubes_in=1, n_cubes_out=1)
    
    # check available gpus
    N_GPU = len(get_available_gpus())

    if N_GPU > 1:
        #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        NUM_WORKERS = N_GPU

        N_BATCH = params['batch_size'] * NUM_WORKERS

        #with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        model = build_compile(model, params, N_GPU)

        #print("Training using multiple GPUs..")

    else:
        NUM_WORKERS = multiprocessing.cpu_count() // 5
        N_BATCH = int(params['batch_size'])

        model = build_compile(model, params, N_GPU)
        #print("Training using single GPU..")


    # load data 
    x_path = '/mnt/home/tmakinen/ceph/data_ska/pca3/'
    y_path = '/mnt/home/tmakinen/ceph/data_ska/'
    workers = multiprocessing.cpu_count() // 5
    train_start = 0
    train_stop = 30
    train_generator = dataloaders.dataLoader3D_static(x_path, y_path, 
                        batch_size=N_BATCH, start=train_start, 
                        stop=train_stop)

    val_start = 80
    val_stop = 90
    val_generator = dataloaders.dataLoader3D_static(x_path, y_path, 
                        batch_size=N_BATCH, start=val_start, 
                        stop=val_stop)

    t1 = time.time()

    # DEFINE CALLBACKS
    # create checkpoint method to save model in the event of walltime timeout
    ## LATER: modify to compute 2D power spectrum

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
    #                         patience=10, min_lr=0.0001, verbose=1)

    N_EPOCHS = 20
    history = model.fit(train_generator,
                                          epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=False, workers=workers)


    return np.min(np.array(history.history['val_mse']))