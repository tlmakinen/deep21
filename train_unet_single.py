
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
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load

# for parallelizing slurm jobs
import os, sys
# import relevant unet model
from unet import unet_models as un
#from unet import data_generators.EpochDataGenerator as EpochDataGenerator

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type  == "GPU"]


if __name__ == '__main__':

	# build model
	model = un.unet_3Layer()
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
								loss="mse",metrics=["mse"])

	# create output directory
	if not os.path.exists('unet_results/'): 
    		os.mkdir('unet_results/')

	# load train / test data
	x_train = np.load('/mnt/home/tmakinen/ceph/ska_sims/pca3_nnu30_train.npy')
	y_train = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nnu30_train.npy")

	x_val = np.load('/mnt/home/tmakinen/ceph/ska_sims/pca3_nnu30_val.npy')
	y_val = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nnu30_val.npy")

	# train the model
	N_EPOCHS = int(sys.argv[1])
	N_BATCH = 192              # big batch size for multiple gpus  
	t1 = time.time()

	# train model
	history = model.fit(x_train,y_train,batch_size=N_BATCH,epochs=N_EPOCHS,validation_data=(x_val, y_val), workers=4, verbose=1)

	t2 = time.time()

	print('total training time for ', N_EPOCHS, ' epochs : ', t2-t1)

	# save the results of the training
	# make outdirectories
	model_fname = sys.argv[2]
	history_fname = sys.argv[3]	

	outfile = "./unet_results/" + model_fname
	model.save(outfile)

	# pickle the training history object
	outfile = "./unet_results/" + history_fname	
	with open(outfile, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	file_pi.close()
