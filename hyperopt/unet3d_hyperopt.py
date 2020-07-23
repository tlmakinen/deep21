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
    'bin_min'       : 1,
    'bin_max'       : 161,
    'nu_start'      : 1,
    'nu_skip'       : 5,
    'nu_dim'        : 32,
    'n_filters'     : 32,
    'conv_width'    : 3,
    'network_depth' : 5,
    'batch_size'    : 48,
    'num_epochs'    : 20,
    'act'           : 'relu', #tf.keras.layers.PReLU(),
    'lr'            : 0.0001, #0.005647691873692045,
    'batchnorm_in'  : True,
    'batchnorm_out' : False,
    'batchnorm_up'  : False,
    'batchnorm_down': True,
    'momentum'      :  0.021165395601698535,
    'model_num'     : int(sys.argv[1]),
    'data_path'     : '/mnt/home/tmakinen/ceph/pca_ska/avg/',
    'nu_indx'       : None,
    'load_model'    : False,
    'noise_level'   : None
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
 
    model.compile(optimizer=tfa.optimizers.Adam(learning_rate=params['lr'], 
                            beta_1=0.9, beta_2=0.999, amsgrad=False), loss="mse",metrics=["mse", custom_loss])
    return model

########################################################################################################################

# MAIN TRAINING FUNCTION
def train_unet(params):
    # initialize model
    model = unet_3d.unet3D(n_filters=params['n_filters'], 
                           nu_dim=configs['nu_dim'],
                           conv_width=params['conv_width'],
                           network_depth=params['network_depth'], 
                           batchnorm_down=params['batchnorm'],
                           batchnorm_in=params['batchnorm'], 
                           batchnorm_out=params['batchnorm'],
                           batchnorm_up=params['batchnorm'], 
                           momentum=params['momentum'])
    
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
    workers = 4
    train_generator = dataloaders.dataLoaderDeep21(
                        path,
                        bin_min=configs['bin_min'],
                        bin_max=configs['bin_max'],
                        is_3d=True, data_type='train',
                        batch_size=N_BATCH, num_sets=3,
                        nu_skip=configs['nu_skip'],
                        sample_size=sample_size,
                        stoch=True,
                        aug=True)

    sample_size=8
    val_generator = dataloaders.dataLoaderDeep21(
                        path,
                        bin_min=configs['bin_min'],
                        bin_max=configs['bin_max'],
                        is_3d=True, data_type='val',
                        batch_size=N_BATCH, num_sets=3,
                        nu_skip=configs['nu_skip'],
                        sample_size=sample_size,
                        stoch=True,
                        aug=True)


    N_EPOCHS = configs['num_epochs']
    history = model.fit(train_generator, epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=False)

    # save transfer function computations
    return np.min(np.array(history.history['val_mse']))

########################################################################################################################

