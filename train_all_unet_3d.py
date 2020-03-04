
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
	N_CUBES = 1
	models = [unet_3d.build_unet3d(n_filters=16, n_cubes=N_CUBES, x_dim=32), unet_3d.build_unet3d_3conv(n_filters=16, n_cubes=N_CUBES, x_dim=32)]
	out_dirs = ['unet3d_4layer_vanilla/', 'unet3d_3layer_vanilla/']


	# build model for this particular job
	model = models[int(sys.argv[1]) - 1]
	out_dir = out_dirs[int(sys.argv[1]) - 1]

	# build model
	# model = unet_3d.build_unet3d()

	N_GPU = 2
	N_GPU = len(get_available_gpus())

	print('-'*10, 'now training unet on ', N_GPU, ' GPUs, output writing to ', out_dir, '-'*10)

	try:
			model = keras.utils.multi_gpu_model(model, gpus=N_GPU)
			print("Training using multiple GPUs..")
	except:
			print("Training using single GPU or CPU..")

	# compile model with specified loss and optimizer
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True), 
								loss="mse",metrics=["mse"])

	# create output directory
	if not os.path.exists(out_dir): 
    		os.mkdir(out_dir)

	## train the model
	N_EPOCHS = int(sys.argv[2])
	N_BATCH = int(768*N_GPU / 2 / N_CUBES)          # every data file has 3 cubes in it => want GD every 1/2 sky => batchsize = 768 / 3 /2


	# load train / test data
	train_path = '/mnt/home/tmakinen/ceph/data_ska/'
	val_path = '/mnt/home/tmakinen/ceph/data_ska/'

	# train_generator = tomo_dataLoader2(train_path, method='train', batch_size=N_BATCH, data_fraction=1., n_cubes=N_CUBES,
	# 								x_dim=(32,32,32,N_CUBES), y_dim = (32,32,32,N_CUBES),)
	# val_generator = tomo_dataLoader2(val_path, method='val', batch_size=N_BATCH, data_fraction=1., n_cubes=N_CUBES,
	# 										x_dim=(32,32,32,N_CUBES), y_dim=(32,32,32,N_CUBES))

	# pull in npy files for the time being
	wins_per_sim = 768  # for (32,32,32) input data
	x = np.load('/mnt/home/tmakinen/ceph/data_ska/pca_3_nnu32_nsim100.npy')[:-2*wins_per_sim]
	y = np.load('/mnt/home/tmakinen/ceph/data_ska/cosmo_nnu032_100sim.npy')[:-2*wins_per_sim]

	# out of 100 total sims
	x_train = x[:78*wins_per_sim]
	x_val = x[-20*wins_per_sim:]
	y_train = y[:78*wins_per_sim]
	y_val = y[-20*wins_per_sim:]

	# expand dims to 4D tensor
	x_train = np.expand_dims(x_train, axis=-1)
	y_train = np.expand_dims(y_train, axis=-1)
	x_val = np.expand_dims(x_val, axis=-1)
	y_val = np.expand_dims(y_val, axis=-1)


	# train the model
	N_EPOCHS = int(sys.argv[2])
	N_BATCH =  768*N_GPU              # big batch size for multiple gpus  


	t1 = time.time()

	# DEFINE CALLBACKS

	# create checkpoint method to save model in the event of walltime timeout
	## LATER: modify to compute 2D power spectrum
	best_fname = out_dir + 'best_model.h5'
	checkpoint = ModelCheckpoint(best_fname, monitor='mse', verbose=0,
    										save_best_only=True, mode='auto', save_freq=15)


	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                              				patience=10, min_lr=0.0001, verbose=1)
	# train model
	# history = model.fit_generator(train_generator,epochs=N_EPOCHS,
	# 									validation_data=val_generator, workers=4, use_multiprocessing=False)
	# 									#, callbacks=[checkpoint])

	history = model.fit(x_train,y_train,batch_size=48,epochs=N_EPOCHS,
							validation_data=(x_val, y_val), workers=2)#, callbacks=[checkpoint])

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

