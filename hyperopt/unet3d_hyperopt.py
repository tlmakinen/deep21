# optimize hyperparameters for 3D UNet
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
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
########################################################################################################################

# DEFINE INPUT PARAMS
configs = {
    'n_filters' : 16,
    'n_cubes_in': 1,
    'n_cubes_out': 1,
    'conv_width' : 2,
    'network_depth': 5,
    'batch_size' : 48,
    'num_epochs' : 30,
    'act' : 'relu',
    'lr': 0.0001,
    'batchnorm_in': True,
    'batchnorm_out': False,
    'batchnorm_up': False,
    'batchnorm_down': True,
    'momentum': 0.02,
    'data_path': '/mnt/home/tmakinen/ceph/data_ska/nu_bin_2/',
    'nu_indx': None,
    'load_model': False,
    'noise_level': None
}
########################################################################################################################
import tensorflow.keras.backend as K
def custom_loss(y_true, y_pred):
   sig = K.mean(K.std(y_true - y_pred))
   return K.log(sig) + (keras.metrics.mse(y_true, y_pred) / (2*K.square(sig))) + 10


########################################################################################################################

# HELPER FUNCTIONS
def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]

def build_compile(net, params, N_GPU):

    model = net.build_model()    
 
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=params['lr'], weight_decay=params['wd'], 
                            beta_1=0.9, beta_2=0.999, amsgrad=False), loss="mse",metrics=["mse", custom_loss])
    return model

########################################################################################################################

# MAIN TRAINING FUNCTION
def train_unet(params):
    # initialize model
    model = unet_3d.unet3D(n_filters=params['n_filters'], conv_width=params['conv_width'],
                        network_depth=params['network_depth'], batchnorm_down=params['batchnorm'],
                        batchnorm_in=params['batchnorm'], batchnorm_out=params['batchnorm'],
                        batchnorm_up=params['batchnorm'], momentum=params['momentum'],
                        n_cubes_in=1, n_cubes_out=1, weight_decay=params['l2_wd'])
    
    # check available gpus
    N_GPU = len(get_available_gpus())

    if N_GPU > 1:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        NUM_WORKERS = N_GPU
        N_BATCH = params['batch_size'] * NUM_WORKERS

        with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
            model = build_compile(model, params, N_GPU)

        print("Training using multiple GPUs..")

    else:
        NUM_WORKERS = 1
        N_BATCH = params['batch_size']

        model = build_compile(model, params, N_GPU)
        print("Training using single GPU..")


    # load data 
    path = configs['data_path']
    sample_size = 20
    train_generator = dataloaders.dataLoaderDeep21(path, 
                        is_3d=True, data_type='train', 
                        batch_size=N_BATCH, num_sets=3,
                        sample_size=sample_size,
                        stoch=True,
                        aug=True)

    sample_size=5
    val_generator = dataloaders.dataLoaderDeep21(path, 
                        is_3d=True, data_type='val', 
                        batch_size=N_BATCH, num_sets=3,
                        sample_size=sample_size,
                        stoch=True,
                        aug=True)



    N_EPOCHS = configs['num_epochs']
    history = model.fit(train_generator, epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=False)

    # save transfer function computations
    return np.min(np.array(history.history['val_mse']))

########################################################################################################################

