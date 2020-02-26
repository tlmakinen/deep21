
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

# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_3d
from unet import tomo_dataLoader2

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]


if __name__ == '__main__':

	# build model
	model = unet_3d.build_unet3d(n_cubes=1)

	N_GPU = 2
	N_GPU = len(get_available_gpus())
	print('num gpu:', N_GPU)
	try:
			model = keras.utils.multi_gpu_model(model, gpus=N_GPU)
			print("Training using multiple GPUs..")
	except:
			print("Training using single GPU or CPU..")
	# compile model with specified loss and optimizer
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True), 
								loss="mse",metrics=["mse"])

	# create output directory
	if not os.path.exists('unet3D_results/'): 
    		os.mkdir('unet3D_results/')

	## train the model
	n_cubes = 3
	N_EPOCHS = int(sys.argv[1])
	N_BATCH = 768*N_GPU / 2 / n_cubes          # every data file has 3 cubes in it => want GD every 1/2 sky => batchsize = 768 / 3 /2


	# load train / test data
	train_path = '/mnt/home/tmakinen/ceph/data_ska/'
	val_path = '/mnt/home/tmakinen/ceph/data_ska/'

	train_generator = tomo_dataLoader2(train_path, method='train', batch_size=N_BATCH, data_fraction=1.,
									x_dim=(32,32,32,3), y_dim = (32,32,32,n_cubes),)
	val_generator = tomo_dataLoader2(val_path, method='val', batch_size=N_BATCH, data_fraction=1.,
											x_dim=(32,32,32,3), y_dim=(32,32,32,n_cubes))

	t1 = time.time()

	# create checkpoint method to save model in the event of walltime timeout
	## LATER: modify to compute 2D power spectrum
	checkpoint = ModelCheckpoint("./unet3D_results/best_model.h5", monitor='loss', verbose=0,
    										save_best_only=True, mode='auto', save_freq=5)

	# train model
	history = model.fit_generator(train_generator,epochs=N_EPOCHS,
										validation_data=val_generator, workers=8, use_multiprocessing=True, callbacks=[checkpoint])

	t2 = time.time()

	print('total training time for ', N_EPOCHS, ' epochs : ', t2-t1)

	# save the results of the training
	# make outdirectories
	model_fname = 'model.h5'
	history_fname = 'history'
	weights_fname = "weights.h5"


	outfile = "./unet3D_results/" + model_fname
	model.save(outfile)

	# save weights
	outfile = "./unet3D_results/" + weights_fname
	model.save_weights(outfile)

	# pickle the training history object
	outfile = "./unet3D_results/" + history_fname	
	with open(outfile, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	file_pi.close()
