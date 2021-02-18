
# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

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
import json
# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_3d
from data_utils import dataloaders, my_callbacks


########################################################################################################################

config_file_path = './configs/configs_deep21.json'

with open(config_file_path) as f:
        configs = json.load(f)



########################################################################################################################

# DEFINE INPUT PARAMS
run_params = {
    'model_num'     : int(sys.argv[1]),
    'load_model'    : False,
}

params = configs['unet_params']
pca_params = configs['pca_params']  
# update parameter dict
params.update(run_params)

########################################################################################################################
import tensorflow.keras.backend as K
def custom_loss(y_true, y_pred):
   sig = K.mean(K.std(y_true - y_pred)) + 1e-6  # for numerical stability
   return K.log(sig) + (keras.metrics.mse(y_true, y_pred) / (2*K.square(sig))) + 10

def custom_loss2(y_true, y_pred):
    return None

########################################################################################################################

# HELPER FUNCTIONS
def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]

def build_compile(net, params, N_GPU):

    if N_GPU > 1:

        if params['load_model']:
            print('loading weights')
            model = net.build_model()
            model.load_weights(out_dir + 'best_weights_{}.h5'.format(params['model_num']))
            #model = keras.utils.multi_gpu_model(model, gpus=N_GPU)

        else:
            #model = keras.utils.multi_gpu_model(net.build_model(), gpus=N_GPU)
            model = net.build_model()
    else:
        if params['load_model']:
            print('loading weights')
            model = keras.models.load_model(out_dir + 'best_model_{}.h5'.format(params['model_num']))
        
        else:
            model = net.build_model()

 
   # model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['lr'],
   #                         beta_1=0.9, beta_2=0.999, amsgrad=False), loss="mse",metrics=["mse", custom_loss])
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=params['lr'], weight_decay=params['wd'], 
                                                 beta_1=0.9, beta_2=0.999, amsgrad=False), 
                                                 loss="logcosh",metrics=["mse", "logcosh"])    

    return model

########################################################################################################################

# MAIN TRAINING FUNCTION
def train_unet(params, out_dir):
    # initialize model
    model = unet_3d.unet3D(n_filters=params['n_filters'], 
                           conv_width=params['conv_width'],
                           nu_dim=params['nu_dim'],
                           x_dim=params['x_dim'],
                           network_depth=params['network_depth'], 
                           batchnorm_down=params['batchnorm_down'],
                           batchnorm_in=params['batchnorm_in'], 
                           batchnorm_out=params['batchnorm_out'],
                           batchnorm_up=params['batchnorm_up'], 
                           momentum=params['momentum']
                           )
    
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
        N_BATCH = params['batch_size']

        model = build_compile(model, params, N_GPU)
        print("Training using single GPU..")

    print('-'*10, 'now training unet on ', N_GPU, ' GPUs, output writing to ', out_dir, '-'*10)
    print('\n','-'*10, 'learning within frequency bins %d--%d'%(params['bin_min'], params['bin_max']-1), '-'*10)
    print('\n', '-'*10, 'skipping every %d frequency, starting from nu=%d'%(params['nu_skip'], params['nu_start']), '-'*10)
    # create output directory
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)


    # load data 
    path = params['data_path']
    workers = 4
    sample_size = 70
    train_generator = dataloaders.dataLoaderDeep21(
                        path,
                        bin_min=params['bin_min'], 
                        bin_max=params['bin_max'], 
                        is_3d=True, data_type='train', 
                        batch_size=N_BATCH, num_sets=pca_params['num_sets'],
                        nu_skip=params['nu_skip'],
                        sample_size=sample_size,
                        nwinds=192,
                        stoch=True,
                        aug=True)

    sample_size=9
    val_generator = dataloaders.dataLoaderDeep21(
                        path,
                        bin_min=params['bin_min'],
                        bin_max=params['bin_max'], 
                        is_3d=True, data_type='val', 
                        batch_size=N_BATCH, num_sets=pca_params['num_sets'],
                        nu_skip=params['nu_skip'],
                        sample_size=sample_size,
                        nwinds=192,
                        stoch=True,
                        aug=True)


    # DEFINE CALLBACKS  
    # create checkpoint method to save model in the event of walltime timeout
    ## LATER: modify to compute 2D power spectrum
    best_fname = out_dir + 'best_model_%d.h5'%(params['model_num'])
    model_checkpoint = ModelCheckpoint(best_fname, monitor='val_mse', verbose=0,
                                        save_best_only=True, mode='auto', save_freq='epoch')
    best_fname = out_dir + 'best_weights_%d.h5'%(params['model_num'])
    weight_checkpoint = ModelCheckpoint(best_fname, monitor='val_mse', verbose=0, save_weights_only=True,
                                        save_best_only=True, mode='auto', save_freq='epoch')


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                             patience=10, min_lr=1e-15, verbose=1)

    clr = my_callbacks.CyclicLR(base_lr=1e-9, max_lr=params['lr'],
                       step_size=560, mode='exp_range', gamma=0.99994)

    #transfer = my_callbacks.transfer(val_generator, 10, batch_size=N_BATCH, patience=1)

    N_EPOCHS = params['num_epochs']
    history = model.fit(train_generator, epochs=N_EPOCHS,validation_data=val_generator, 
                                          use_multiprocessing=True, workers=workers, 
                                          callbacks=[reduce_lr, model_checkpoint, weight_checkpoint])
    #test = np.concatenate([np.expand_dims(
    #                         np.load('/mnt/home/tmakinen/ceph/data_ska/bin1/test/pca3_sim%03d.npy'%(i+1)), 
                                                                       #    axis=-1) for i in range(90, 100)])
    # make test  predictioni
    #y_pred = model.predict(test, batch_size=N_BATCH)
    
    # save transfer function computations
    return history, model#, y_pred

########################################################################################################################

if __name__ == '__main__':

    # create output directoryi
    out_dir = params['out_dir'] +  'unet_results_{}_{}/'.format(params['bin_min'], params['bin_max'])
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    # make specific output dir for model #
    #out_dir = params['out_dir']
    

    t1 = time.time()

    history,model = train_unet(params, out_dir)

    t2 = time.time()

    print('total training time for ', params['num_epochs'], ' epochs : ', (t2-t1) / (60*60), ' hours')

    # save the results of the training
    # make outdirectories
    model_fname = 'model_{}.h5'.format(params['model_num'])
    history_fname = 'history_{}'.format(params['model_num'])
    weights_fname = "weights_{}.h5".format(params['model_num'])
    #transfer_fname = 'transfer'
    if params['load_model']:
        history_fname += '_continued'
        #transfer_fname += '_continued'

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
    # outfile = out_dir + transfer_fname	
    # with open(outfile, 'wb') as file_pi:
    #     pickle.dump(transfer, file_pi)
    # file_pi.close()

    # compute y_pred using the best model weights
    #outfile = out_dir + 'y_pred'
    #np.save(outfile, y_pred)

    # save all model params for later reference
    outfile = out_dir + 'params'
    with open(outfile, 'wb') as file_pi:
        pickle.dump(params, file_pi)
    file_pi.close()
