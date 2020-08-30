
# script for analyzing nn performance on different noise levels
# import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import healpy as hp
import h5py
from scipy import fftpack
import sys,os
import time

from analysis_library import *
###############################################################################

################################################################################ Define all input and output dirs

# inputs for data and models to test
model_path = '/mnt/home/tmakinen/ceph/deep21_results/unpolarized/unet_results_1_193/'
data_path = '/mnt/home/tmakinen/ceph/pca_ska/nside4_avg/test_noise'
info_path = '/mnt/home/tmakinen/repositories/deep21/sim_info/'
rearr_file = 'rearr_nside4.npy'


# make outdirs for each 
outdir = '/mnt/home/tmakinen/ceph/deep21_results/unpolarized/noise_test/'


types = ['_plus', '_minus', 'control']


# data parameters
N_NU = 64
NU_AVG = 3
WINDOW_NSIDE = 4
N_WINDS = 192

bin_min = 0
bin_max = 192

num_nets = 9
num_sims = 1
sim_start = 1

# remove mean for radial Pka ?
remove_mean = False

# to store every ensemble member's prediction
mse_arr = []

# loop over all noise realizations
for o in range(1):
   o = int(sys.argv[1]) - 1
   print('working on %d simulations, writing to %s'%(num_sims, outdir))

   pca3 =  np.load(data_path + '_%03d/pca3_sim093.npy'%(o+sim_start))
   pca6 =  np.load(data_path + '_%03d/pca6_sim093.npy'%(o+sim_start))
   cosmo = np.load(data_path + '_%03d/cosmo_sim093.npy'%(o+sim_start))
   noise = np.load(data_path + '_%03d/cosmo_noisy_sim093.npy'%(o+sim_start))

   # make nn prediction
   fname = 'nn_noise_%03d'%(o + sim_start)
   nn_preds = ensemble_prediction(model_path, num_nets, pca3, outfname=outdir + fname)

   # compute ensemble weights
   w_mse = [np.mean(compute_mse(cosmo, n)) for n in nn_preds]
   np.save(outdir + 'ensemble_mse_%d'%(o), np.array(w_mse))
   
   # compute pca-6 mse
   w_mse = np.mean(compute_mse(cosmo, pca6))
   np.save(outdir + 'pca6_mse_%d'%(o), np.array(w_mse))
   #mse_arr.append(np.array(w_mse))       
   del nn_preds

# save output
#np.save(outdir + 'mse_array', np.array(mse_arr))
        

