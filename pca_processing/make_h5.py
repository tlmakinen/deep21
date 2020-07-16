mport time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

t1 = time.time()


# input paths
path = '/mnt/home/tmakinen/ceph/data_ska/'

# for dataset indexing
dat_type = ['test', "train", "val"]

# output paths
output_base = "/mnt/home/tmakinen/ceph/data_ska/bin1/"


# take in inputs and stack 

for j in range(1):
    print(j)
    
    data = np.concatenate([(np.load('/mnt/home/tmakinen/ceph/data_ska/bin1/data_%d/pca3_sim%03d.npy'%(j+1, i+1))) for i in range(100)], axis=0)
    y = np.concatenate([(np.load('/mnt/home/tmakinen/ceph/data_ska/bin1/data_%d/cosmo_sim%03d.npy'%(j+1, i+1))) for i in range(100)], axis=0)

    
    # concatenate x and y on top of one another
    
    data = np.stack((data,y), axis=-1)
    

    # split into train and test sets
    
    num_train = int(len(data)*0.80)
    num_val = int(len(data)*0.1)
    num_test = int(len(data)*0.1)
    
    test_data = data[-num_test:]
    arr = data[:-num_test]
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    arr_list = [np.array(a) for a in [test_data, train_data, val_data]]
    
    # open h5 file
    out_fname = output_base + "dataset_%d"%(j+1) + '.h5'
    h5f = h5py.File(out_fname, 'w')
    
    for k in range(len(dat_type)):  # indexes over train, test, val  
        # don't transpose data so that we have [blah, 32, 32, 32, 2]
        h5f.create_dataset(dat_type[k], data=arr_list[k])
    
    # close the h5 file
    h5f.close()


t2 = time.time()
print('time for h5 arrangement: ', (t2-t1) / 60, ' minutes')
          

