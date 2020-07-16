# data loader script for training unets
from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils.data_utils import Sequence
import h5py
import os

class dataLoaderDeep21(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 path, 
                 data_type='train', 
                 is_3d = True,
                 batch_size=48, 
                 num_sets=1,
                 sample_size=20,
                 shuffle=False,
                 bin_min = 1,
                 bin_max = 160, 
                 nu_indx=None,
                 aug = True,
                 stoch = True
                ):
        
        
        'Initialization'
        self.data_type = data_type
        self.is_3d = is_3d
        self.sample_size = sample_size
        self.num_sets = num_sets
        self.nwinds = 768          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bin_min = bin_min - 1 # python indexes from 0
        self.bin_max = bin_max - 1
        self.path = path
        self.stoch = stoch
        self.aug = aug
        self.fname = path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
        self.datafile = h5py.File(self.fname, 'r')
        self._dat_length = len(self.datafile[self.data_type])
        self.datafile.close()
        self.indexes = np.arange(self.nwinds*self.sample_size)
        self.nu_indx = nu_indx
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(((self.sample_size)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        #fname = self.path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
        #ind = self.indexes[idx]  # index in terms of shuffled data
        x,y = self.load_data(idx)
        
        return x,y
    
    def on_epoch_end(self):
        # switch up dataset every other time to change noise
        if np.random.rand() > 0.5:
           # self.datafile.close()
            self.fname = self.path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
           # self.datafile = h5py.File(self.fname, 'r')

        # draw randomly with replacement but keep dataset fixed
        if self.stoch:
            # now shuffle the dataset indices in chunks of self.nwinds:
            inds = np.arange(self._dat_length).reshape(-1, 768)
            l = np.arange(len(inds))
            # choose random skies with replacement
            inds = inds[np.random.choice(l, size=self.sample_size)]
            # reshape array
            inds = inds.reshape(self.sample_size*self.nwinds)
            self.indexes = inds



    
    
    def load_data(self, idx):
        self.datafile = h5py.File(self.fname, 'r')
        d = self.datafile[self.data_type][self.indexes[idx*self.batch_size:(idx+1)*self.batch_size], :, :, self.bin_min:self.bin_max, : ]
        x = d.T[0].T
        y = d.T[1].T; del d
        self.datafile.close()
        
        # flip boxes according to random draw:
        r = np.random.rand()
        
        if self.aug:        
            # random image rotations in x and y
            if r < 0.25:
                x = np.rot90(x, k=1, axes=(1,2))
                y = np.rot90(y, k=1, axes=(1,2))

            elif r < 0.50:
                x = np.rot90(x, k=-1, axes=(1,2))
                y = np.rot90(y, k=-1, axes=(1,2))
            elif r < 0.75:
                x = np.rot90(x, k=-2, axes=(1,2))
                y = np.rot90(y, k=-2, axes=(1,2))	

        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            x = x.T[self.nu_indx].T       
            y = y.T[self.nu_indx].T   
        
        if self.is_3d:
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)

        return x,y

class dataLoader3D(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 x_path, 
                 y_path, 
                 data_type='train', 
                 batch_size=48, 
                 start=1, stop=90,
                 shuffle=False, 
                 nu_indx=None
                    ):
        
        
        'Initialization'
        self.data_type = data_type + '_data'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.y_fname = y_path + 'cosmo.h5'    # y filename doesn't change

        self.x_path = x_path
        self.x_fname = self.x_path 


    def __len__(self):
        return int(np.floor(((self.stop-self.start)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        x_fname = self.x_path +  'pca3_%d.h5'%(int(np.ceil(np.random.rand()*5)))
 
        batch_x = self.load_data(x_fname, idx)
        batch_y = self.load_data(self.y_fname, idx)
        
        return batch_x, batch_y
    
    def __gettruth__(self):
        return h5py.File(self.y_fname, 'r')[self.data_type][self.start*self.nwinds:self.stop*self.nwinds]

    
    def load_data(self, fname, idx):
        d = h5py.File(fname, 'r')[self.data_type]
        d = d[idx * self.batch_size:(idx + 1) * self.batch_size]
        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T        
        d = np.expand_dims(d, axis=-1)
        return d



class dataLoader3D_static(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_path, y_path, batch_size=48, 
                    num_sets = 3,
                    start=0, stop=80,
                    shuffle=False, 
                    nu_indx=None, 
                    aug=True):
        
        
        'Initialization'
        self.start = start
        self.stop = stop
        self.aug = aug
        self.nwinds = 768          # simulation param, num bricks per sim
        self.y_fnames = [y_path + 'cosmo/cosmo_nnu032_sim%03d.npy'%(i+1) for i in range(start, stop)]
        self.x_fnames = [x_path + '_%d/pca_3comp_nnu32_sim%03d.npy'%(int(np.ceil(np.random.rand()*num_sets)),i+1) for i in range(start, stop)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.x_data = np.concatenate([self.load_data(p) for p in self.x_fnames], axis=0)
        self.y_data = np.concatenate([self.load_data(p) for p in self.y_fnames], axis=0)


    def __len__(self):
        return int(np.floor(len(self.x_data) // self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
    
    def __gettruth__(self):
        return np.squeeze(self.y_data)

    
    def load_data(self, fname):
        d = np.load(fname)
        # shuffle frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T
        
        d = np.expand_dims(d, axis=-1)
        return d


class dataLoader3D_obs(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_path, y_path, batch_size=48, 
                    num = 50, num_sets=1,
                    start=0, stop=80,
                    shuffle=False, nu_indx=None):
        
        
        'Initialization'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.y_fnames = [y_path + 'cosmo/cosmo_nnu032_sim%03d.npy'%(i+1) for i in range(start, stop)]
        self.x_dir = os.listdir(x_path + '_1')
        self.x_fnames = [ x_path + '_%d/'%(int(np.ceil(np.random.rand()*num_sets))) + 
                                                        self.x_dir[i+1] for i in range(start, stop)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        # log the inputs
        self.x_data = np.concatenate([self.load_data(p) for p in self.x_fnames], axis=0)
        self.x_data = np.log(self.x_data + 0.1 + np.abs(np.min(self.x_data)))
        # outputs
        self.y_data = np.concatenate([self.load_data(p) for p in self.y_fnames], axis=0)


    def __len__(self):
        return int(np.floor(len(self.x_data) // self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
    
    def __gettruth__(self):
        return np.squeeze(self.y_data)

    
    def load_data(self, fname):
        d = np.load(fname)
        # shuffle frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T
        
        d = np.expand_dims(d, axis=-1)
        return d




class dataLoader2D(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 x_path, 
                 y_path, 
                 data_type='train', 
                 batch_size=48, 
                 start=1, stop=90,
                 shuffle=False, 
                 nu_indx=None
                    ):
        
        
        'Initialization'
        self.data_type = data_type + '_data'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.y_fname = y_path + 'cosmo.h5'    # y filename doesn't change

        self.x_path = x_path
        self.x_fname = self.x_path 


    def __len__(self):
        return int(np.floor(((self.stop-self.start)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        x_fname = self.x_path +  'pca3_%d.h5'%(int(np.ceil(np.random.rand()*5)))
 
        batch_x = self.load_data(x_fname, idx)
        batch_y = self.load_data(self.y_fname, idx)
        
        return batch_x, batch_y
    
    def __gettruth__(self):
        return h5py.File(self.y_fname, 'r')[self.data_type][self.start*self.nwinds:self.stop*self.nwinds]

    
    def load_data(self, fname, idx):
        d = h5py.File(fname, 'r')[self.data_type]
        d = d[idx * self.batch_size:(idx + 1) * self.batch_size]
        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T        
        #d = np.expand_dims(d, axis=-1)
        return d



class dataLoader2D_static(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_path, y_path, batch_size=48, 
                    start=1, stop=20, num_sets=5,
                    shuffle=False, nu_indx=None):
        
        
        'Initialization'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.y_fnames = [y_path + 'cosmo/cosmo_nnu032_sim%03d.npy'%(i+1) for i in range(start, stop)]
        self.x_fnames = [x_path + 'pca3_%d/pca_3comp_nnu32_sim%03d.npy'%(int(np.ceil(np.random.rand()*num_sets)),i+1) for i in range(start, stop)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.x_data = np.concatenate([self.load_data(p) for p in self.x_fnames], axis=0)
        self.y_data = np.concatenate([self.load_data(p) for p in self.y_fnames], axis=0)


    def __len__(self):
        return int(np.floor(len(self.x_data) // self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
    
    def __gettruth__(self):
        return np.squeeze(self.y_data)

    
    def load_data(self, fname):
        d = np.load(fname)
        # shuffle frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T
        
        return d