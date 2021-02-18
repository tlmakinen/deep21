import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys,os
import json

t1 = time.time()

# OPEN CONFIG FILE
config_file_path = sys.argv[2] + 'configs_deep21.json'

with open(config_file_path) as f:
        configs = json.load(f)

dir_configs = configs["directory_configs"]
pca_configs = configs["pca_params"]
unet_configs = configs["unet_params"]


# which dataset we're working with
j = int(sys.argv[1]) - 1

# data shape
data_shape = (pca_configs["N_WINDS"], unet_configs["x_dim"], 
              unet_configs["x_dim"], unet_configs["nu_dim"], 2)

# input paths
path = dir_configs["data_path"] + "data"

# for dataset indexing
dat_type = ['test', "train", "val"]
num_data = pca_configs["N_SIMS"]
# split into train and test sets

num_train = int(num_data*0.80)
num_val = int(num_data*0.1)
num_test = int(num_data*0.1)



# output paths
output_base = dir_configs["data_path"]



# open h5 file
out_fname = output_base + "dataset_%d"%(j+1) + '.h5'
h5f = h5py.File(out_fname, 'w')


print('data shape : ', data_shape)

# make training dataset
print('assembling training data for dataset {}'.format(j+1))
dset = h5f.create_dataset(name='train', shape=data_shape, maxshape=(None,64,64,64,2))
for l in range(num_train):
    # load next sim. take inputs and stack vertically along last axis
    data = np.load(path + '_%d/pca3_sim%03d.npy'%(j+1, l+1))
    y = np.load(path + '_%d/cosmo_sim%03d.npy'%(j+1, l+1))
    data = np.stack((data,y), axis=-1); del y

    dset[-data.shape[0]:] = data
    # resize the dataset to accept new data
    dset.resize(dset.shape[0]+data.shape[0], axis=0)

    print('big dataset shape: ', dset.shape)


# make val dataset
dset = h5f.create_dataset(name='val', shape=data_shape, maxshape=(None,64,64,64,2))
for l in range(num_train, num_train+num_val):
    print('assembling validation data')
    # load next sim
    data = np.load(path + '_%d/pca3_sim%03d.npy'%(j+1, l+1))
    y = np.load(path + '_%d/cosmo_sim%03d.npy'%(j+1, l+1))
    data = np.stack((data,y), axis=-1); del y
    dset[-data.shape[0]:] = data
    # resize dataset to accept new data
    dset.resize(dset.shape[0]+data.shape[0], axis=0)

    print('big dataset shape: ', dset.shape)

h5f.close()




t2 = time.time()
print('time for h5 arrangement: ', (t2-t1) / 60, ' minutes')
          

