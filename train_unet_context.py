
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

# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_3d
from unet import tomo_dataLoader2




def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]


if __name__ == '__main__':

# HOW IS THE DATA STACKED ?
    N_CUBES = 2
    model = unet_3d.unet3d_con(n_filters=16, n_cubes_in=N_CUBES, activation='relu',
                                    n_cubes_out=1, x_dim=32, momentum=0.1)

    out_dir = './contextnet_3d/'
    # create output directory
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    N_GPU = len(get_available_gpus())

    print('-'*10, 'now training unet on ', N_GPU, ' GPUs', '-'*10)

    try:
            model = keras.utils.multi_gpu_model(model, gpus=N_GPU)
            print("Training using multiple GPUs..")
    except:
            print("Training using single GPU or CPU..")

    # compile model with specified loss and optimizer
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True), 
                            loss="mse",metrics=["mse"])


    # load train / test data
    train_path = '/mnt/home/tmakinen/ceph/data_ska/'
    val_path = '/mnt/home/tmakinen/ceph/data_ska/'

    # pull in npy files for the time being
    wins_per_sim = 768  # for (32,32,32) input data

    x_train = h5py.File(train_path + 'obs32.h5', 'r')['train_data'][:]#[:768*80]
    y_train = h5py.File(train_path + 'cosmo32.h5', 'r')['train_data'][:]#[:768*80]

    x_val = h5py.File(train_path + 'obs32.h5', 'r')['val_data']#[768*80:768*90]
    y_val = h5py.File(train_path + 'cosmo32.h5', 'r')['val_data']#[768*80:768*90]


    context_path = train_path + 'context/'
    brick = np.load('/mnt/home/tmakinen/ceph/data_ska/context/fg_2context_nnu032_sim001.npy')
    fg = brick.T[::2].T
    con = brick.T[1::2].T

    for i in range(1, 90):
        brick2 = np.load('/mnt/home/tmakinen/ceph/data_ska/context/fg_2context_nnu032_sim%03d.npy'%(i))
        fg2 = brick2.T[::2].T
        con2 = brick2.T[1::2].T
        con = np.concatenate((con, con2), axis=0)
        fg = np.concatenate((fg, fg2), axis=0)

    con_train = con[:768*80]
    con_val = con[768*80:768*90]
    fg_train= fg[:768*80]
    fg_val = fg[768*80:768*90]

    # assemble inputs
    input_train = np.array([x_train.T, con_train.T]).T
    input_val = np.array([x_val.T, con_val.T]).T

    output_train = np.array([y_train.T]).T
    output_val = np.array([y_val.T]).T

    # train the model
    N_EPOCHS = 250 #int(sys.argv[2])
    N_BATCH =  96             # big batch size for multiple gpus  


    t1 = time.time()

    # learning rate scheduler and callbacks 
    best_fname = out_dir + 'best_model.h5'
    checkpoint = ModelCheckpoint(best_fname, monitor='mse', verbose=0,
                                                    save_best_only=True, mode='auto', save_freq=15)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                            patience=7, min_lr=0.00001, verbose=1)


    history = model.fit(input_train,output_train,batch_size=N_BATCH,epochs=N_EPOCHS,
                    validation_data=(input_val, output_val), workers=2, callbacks=[checkpoint, reduce_lr])

    t2 = time.time()

    print('total training time for ', N_EPOCHS, ' epochs : ', (t2-t1) / (60*60), ' hours')

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
    #model.load_weights(best_fname)
    np.save(outfile, model.predict(x_val))



