
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
from data_utils import dataloaders, my_callbacks

########################################################################################################################

# HELPER FUNCTIONS
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

########################################################################################################################

# DEFINE INPUT PARAMS
params = {
    'n_filters' : 16,
    'n_cubes_in': 1,
    'n_cubes_out': 1,
    'conv_width' : 2,
    'network_depth': 4,
    'batch_size' : 48,
    'num_epochs' : 200,
    'act' : 'relu',
    'lr': 0.0001,
    'batchnorm_in': True,
    'batchnorm_out': False,
    'batchnorm_up': False,
    'batchnorm_down': False,
    'momentum': 0.06,
    'out_dir': 'model_{}/'.format(int(sys.argv[1])),
    'data_path': '/mnt/home/tmakinen/ceph/data_ska/nu_bin_2/',
    'nu_indx': None,
    'load_weights': True,
    'noise_level': None
}


########################################################################################################################

# MAIN TRAINING FUNCTION
def train_unet(params, out_dir):
    # initialize model
    model = unet_3d.unet3D(n_filters=16, conv_width=params['conv_width'],
                        network_depth=params['network_depth'], batchnorm_down=params['batchnorm_down'],
                        batchnorm_in=params['batchnorm_in'], batchnorm_out=params['batchnorm_out'],
                        batchnorm_up=params['batchnorm_up'], momentum=params['momentum'],
                        n_cubes_in=1, n_cubes_out=1)
    
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
        NUM_WORKERS = multiprocessing.cpu_count() // 5
        N_BATCH = params['batch_size']

        model = build_compile(model, params, N_GPU)
        print("Training using single GPU..")

    print('-'*10, 'now training unet on ', N_GPU, ' GPUs, output writing to ', out_dir, '-'*10)


    # create output directory
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)


    # load data 
    path = params['data_path']
    workers = 4
    train_start = 0
    train_stop = 70
    train_generator = dataloaders.dataLoaderDeep21(path, 
                        is_3d=True, data_type='train', 
                        batch_size=N_BATCH, num_sets=3,
				    start=train_start, stop=train_stop,
                        aug=True)

    val_start = 80
    val_stop = 90
    val_generator = dataloaders.dataLoaderDeep21(path, 
                        is_3d=True, data_type='val', 
                        batch_size=N_BATCH, num_sets=3,
				    start=val_start, stop=val_stop,
                        aug=True)


    # DEFINE CALLBACKS  
    # create checkpoint method to save model in the event of walltime timeout
    ## LATER: modify to compute 2D power spectrum
    best_fname = out_dir + 'best_model.h5'
    checkpoint = ModelCheckpoint(best_fname, monitor='val_mse', verbose=0,
                                        save_best_only=True, mode='auto', period=25)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                             patience=4, min_lr=1e-6, verbose=1)

    transfer = my_callbacks.transfer(val_generator, 10, batch_size=N_BATCH, patience=1)

    N_EPOCHS = params['num_epochs']
    history = model.fit(train_generator, epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=False, workers=workers, callbacks=[reduce_lr, transfer, checkpoint])

    # make validation prediction
    y_pred = model.predict(val_generator, workers=4, steps=np.ceil(768*10/N_BATCH))
    
    # save transfer function computations
    return history, model, y_pred, transfer.get_data()

########################################################################################################################

if __name__ == '__main__':

    # create output directory
    if not os.path.exists(params['out_dir']): 
        os.mkdir(params['out_dir'])
        
    # make specific output dir
    out_dir = params['out_dir']


    t1 = time.time()

    history,model,y_pred,transfer = train_unet(params, out_dir)

    t2 = time.time()

    print('total training time for ', params['num_epochs'], ' epochs : ', (t2-t1) / (60*60), ' hours')

    # save the results of the training
    # make outdirectories
    model_fname = 'model.h5'
    history_fname = 'history'
    weights_fname = "weights.h5"
    transfer_fname = 'transfer'
    if params['load_weights']:
        history_fname += '_continued'
        transfer_fname += '_continued'

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

    # pickle the transfer object
    outfile = out_dir + transfer_fname	
    with open(outfile, 'wb') as file_pi:
        pickle.dump(transfer, file_pi)
    file_pi.close()

    # compute y_pred using the best model weights
    outfile = out_dir + 'y_pred'
    np.save(outfile, y_pred)

    # save all model params for later reference
    outfile = out_dir + 'params'
    with open(outfile, 'wb') as file_pi:
        pickle.dump(transfer, file_pi)
    file_pi.close()
