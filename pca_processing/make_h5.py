import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys,os

t1 = time.time()

# which dataset we're working with
j = int(sys.argv[1]) - 1

# data shape
data_shape = (768, 32, 32, 690, 2)

# input paths
path = '/mnt/home/tmakinen/ceph/pca_ska/'

# for dataset indexing
dat_type = ['test', "train", "val"]
num_data = 90
# split into train and test sets

num_train = int(num_data*0.80)
num_val = int(num_data*0.1)
num_test = int(num_data*0.1)



# output paths
output_base = "/mnt/home/tmakinen/ceph/pca_ska/"



# open h5 file
out_fname = output_base + "dataset_%d"%(j+1) + '.h5'
h5f = h5py.File(out_fname, 'w')


print('data shape : ', data_shape)

# make training dataset
print('assembling training data for dataset {}'.format(j))
dset = h5f.create_dataset(name='train', shape=data_shape, maxshape=(None,32,32,690,2))
for l in range(num_train):
    # load next sim. take inputs and stack vertically along last axis
    data = np.load('/mnt/home/tmakinen/ceph/pca_ska/data_%d/pca3_sim%03d.npy'%(j+1, l+1))
    y = np.load('/mnt/home/tmakinen/ceph/pca_ska/data_%d/cosmo_sim%03d.npy'%(j+1, l+1))
    data = np.stack((data,y), axis=-1); del y

    dset[-data.shape[0]:] = data
    # resize the dataset to accept new data
    dset.resize(dset.shape[0]+data.shape[0], axis=0)

    print('big dataset shape: ', dset.shape)



# make val dataset
dset2 = h5f.create_dataset(name='val', shape=data_shape, maxshape=(None,32,32,690,2))
for l in range(78, 78+num_val):
    print('assembling validation data')
    # load next sim
    data = np.load('/mnt/home/tmakinen/ceph/pca_ska/data_%d/pca3_sim%03d.npy'%(j+1, l+1))
    y = np.load('/mnt/home/tmakinen/ceph/pca_ska/data_%d/cosmo_sim%03d.npy'%(j+1, l+1))
    data = np.stack((data,y), axis=-1); del y
    dset[-data.shape[0]:] = data
    # resize dataset to accept new data
    dset.resize(dset.shape[0]+data.shape[0], axis=0)

    print('big dataset shape: ', dset.shape)

h5f.close()




t2 = time.time()
print('time for h5 arrangement: ', (t2-t1) / 60, ' minutes')
          

