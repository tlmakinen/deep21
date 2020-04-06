
# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

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
    if params['load_weights']:
        model = keras.models.load_model(out_dir + 'model.h5')
    else:
        model = net.build_model()
    if N_GPU > 1:
        model = keras.utils.multi_gpu_model(model, gpus=N_GPU)

 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr'], 
                            beta_1=0.9, beta_2=0.999, amsgrad=True), loss="mse",metrics=["mse"])
    return model

# DEFINE INPUT PARAMS
params = {
    'n_filters' : 16,
    'n_cubes_in': 1,
    'n_cubes_out': 1,
    'conv_width' : 1,
    'network_depth': int(sys.argv[1]),
    'batch_size' : 48,
    'num_epochs' : 200,
    'act' : 'relu',
    'lr': 0.0001,
    'out_dir': 'trials/',
    'nu_indx': np.arange(32),
    'load_weights': False
    }

def train_unet(params, out_dir):
    # initialize model
    model = unet_3d.unet3D(n_filters=params['n_filters'], conv_width=params['conv_width'],
                        network_depth=params['network_depth'], n_cubes_in=params['n_cubes_in'], 
                                                                    n_cubes_out=params['n_cubes_out'])
    
    # check available gpus
    N_GPU = len(get_available_gpus())

    if N_GPU > 1:
        #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        NUM_WORKERS = N_GPU

        N_BATCH = params['batch_size'] * NUM_WORKERS

        #with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        model = build_compile(model, params, N_GPU)

        print("Training using multiple GPUs..")

    else:
        NUM_WORKERS = multiprocessing.cpu_count() // 5
        N_BATCH = params['batch_size']

        model = build_compile(model, params, N_GPU)
        print("Training using single GPU..")

    print('-'*10, 'now training unet on ', N_GPU, ' GPUs, output writing to ', out_dir, '-'*10)


    # create output directory
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)


    # load data 
    x_path = '/mnt/home/tmakinen/ceph/data_ska/'
    y_path = '/mnt/home/tmakinen/ceph/data_ska/'
    workers = multiprocessing.cpu_count() // 5
    train_start = 0
    train_stop = 50
    train_generator = dataloaders.dataLoader3D_static(x_path, y_path, 
                        batch_size=N_BATCH, #data_type='train',
 			start=train_start, 
                        stop=train_stop, nu_indx=params['nu_indx'])

    val_start = 80
    val_stop = 90
    val_generator = dataloaders.dataLoader3D_static(x_path, y_path, 
                        batch_size=N_BATCH, # data_type='val', 
			start=val_start, 
                        stop=val_stop, nu_indx=params['nu_indx'])

    t1 = time.time()

    # DEFINE CALLBACKS
    # create checkpoint method to save model in the event of walltime timeout
    ## LATER: modify to compute 2D power spectrum
    best_fname = out_dir + 'best_model.h5'
    checkpoint = ModelCheckpoint(best_fname, monitor='mse', verbose=0,
                                        save_best_only=True, mode='auto', save_freq=15)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                            patience=10, min_lr=0.0001, verbose=1)

    N_EPOCHS = params['num_epochs']
    history = model.fit(train_generator, epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=False, workers=workers)#, callbacks=[checkpoint])

    # make validation prediction
    y_pred = model.predict(val_generator, workers=5, steps=np.ceil(768*10/N_BATCH))
    # pull out y_true
    y_true = val_generator.__gettruth__()

    return history, model, y_pred, y_true

if __name__ == '__main__':

    # create output directory
    if not os.path.exists(params['out_dir']): 
        os.mkdir(params['out_dir'])
        
    # make specific output dir
    out_dir = params['out_dir'] + 'unet3d_{}conv_{}deep/'.format(params['conv_width'], params['network_depth'])

    # create output directory
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)


    t1 = time.time()

    history,model,y_pred,y_true = train_unet(params, out_dir)

    t2 = time.time()

    print('total training time for ', params['num_epochs'], ' epochs : ', (t2-t1) / (60*60), ' hours')

    # save the results of the training
    # make outdirectories
    model_fname = 'model.h5'
    history_fname = 'history'
    weights_fname = "weights.h5"


    outfile = out_dir + model_fname
    model.save(outfile)

    # save weights
    outfile = out_dir + weights_fname
    model.save_weights(outfile)

    # pickle the training history object
    outfile = out_dir + history_fname	
    with open(outfile, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    file_pi.close()

    # compute y_pred using the best model weights
    outfile = out_dir + 'y_pred'
    np.save(outfile, y_pred)

    outfile = out_dir + 'y_true'
    np.save(outfile, y_true)

