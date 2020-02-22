
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

# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_2d as un
#from unet import data_generators.EpochDataGenerator as EpochDataGenerator

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]


if __name__ == '__main__':

	models = [un.unet_conv4(n_filters=64, n_channels=64), un.unet_conv3(n_filters=64, n_channels=64)]
	out_dirs = ['unet_4layer_vanilla/', 'unet_3layer_vanilla/']


	# build model for this particular job
	model = models[int(sys.argv[1])]
	out_dir = out_dirs[int(sys.argv[1])]
	#model = keras.models.load_model('models_network1/model_full_1')

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
								loss="mse",metrics=["accuracy"])

	# create output directory
	if not os.path.exists(out_dir): 
		os.mkdir(out_dir)

	print('-'*10, 'now training unet, output writing to ', out_dir, '-'*10)

	# load train / val data (cut out test set)
	x = np.load('/mnt/home/tmakinen/ceph/ska_sims/pca_3comp_reduced_mid_nnu64_nsim100.npy')[:-2*192]
	y = np.load('/mnt/home/tmakinen/ceph/ska_sims/cosmo_nnu064_100sim.npy')[:-2*192]


	x_train = x[:78*192]
	x_val = x[-20*192:]

	y_train = y[:78*192]
	y_val = y[-20*192:]

	# train the model
	N_EPOCHS = int(sys.argv[2])
	N_BATCH = 192              # big batch size for multiple gpus  
	t1 = time.time()

	# train model
	history = model.fit(x_train,y_train,batch_size=N_BATCH,epochs=N_EPOCHS,validation_data=(x_val, y_val), workers=4, verbose=1)

	t2 = time.time()

	print('total training time for model', out_dir, 'for ', N_EPOCHS, ' epochs : ', t2-t1)

	# save the results of the training
	# make outdirectories
	#model_fname = sys.argv[3]
	#history_fname = sys.argv[4]	

	# save model (will be multi-gpu model)
	fname = out_dir + 'model.h5'	
	model.save(fname)

	# save model weights
	fname = out_dir + 'weights.h5'
	model.save_weights(fname)

	# make prediction and save y_pred
	fname = out_dir + 'y_pred'
	np.save(fname, model.predict(x_val))

	# pickle the training history object
	outfile = out_dir + 'history'
	with open(outfile, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	file_pi.close()
