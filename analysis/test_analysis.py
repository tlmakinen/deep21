# Script for running analysis on multiple test data
# by TLM 

## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import healpy as hp
import h5py
from scipy import fftpack
import sys,os

from analysis_library import *

###############################################################################

###############################################################################
# Define all input and output dirs

# inputs for data and models to test
model_path = '/mnt/home/tmakinen/jobs2/bignet/results_logp_1_193/'
data_path = '/mnt/home/tmakinen/ceph/pca_ska/nside4_avg/test/'
info_path = '/mnt/home/tmakinen/repositories/deep21/sim_info/'
rearr_file = 'rearr_nside4.npy'


# make outdirs for each 
outdirs = ['/mnt/home/tmakinen/ceph/deep21_results/test_sim_' + str(i+91) + '/' for i in range(10)]


types = ['_plus', '_minus', 'control']


# data parameters
N_NU = 64
NU_AVG = 3
WINDOW_NSIDE = 4
N_WINDS = 192

bin_min = 0
bin_max = 192

if __name__ == '__main__':

    for l,fg_type in enumerate(['plus']):
        
        o = int(sys.argv[1]) - 1 # slurm indexes +1
        outdir = outdirs[o]
        o = o + 91

        print('working on simulation %d, writing to %s'%(o+1, outdir))

        pca3 = np.load(data_path + 'pca3_sim%03d.npy'%(o+1))
        pca6 = np.load(data_path + 'pca6_sim%03d.npy'%(o+1))
        cosmo = np.load(data_path + 'cosmo_sim%03d.npy'%(o+1))
        noise = np.load(data_path + 'cosmo_noisy_sim%03d.npy'%(o+1))

        # make nn prediction
        nn_preds = ensemble_prediction(model_path, 5, pca3, outfname=outdir)

        # compute ensemble weights
        w_logp = [np.mean(compute_logp(cosmo, n)) for n in nn_preds]
        np.save(outdir + 'ensemble_weights', np.array(w_logp))

        # now average maps together according to weights
        ensemble_predicted_map = np.average(nn_preds)

        # compute power spectra 
        ensemble_predicted_Cl = []
        ensemble_residual_Cl = []
        for m,prediction in enumerate(nn_preds):
            cosmo_Cl, nn_pred_Cl, nn_res_Cl = angularPowerSpec(cosmo, prediction, bin_min=bin_min, bin_max=bin_max, rearr=info_path + rearr_file, nu_arr=info_path+'nuTable.txt', 
                                                                    NU_AVG=NU_AVG, N_NU=N_NU, out_dir=outdir + 'angular/', name='nn', save_spec=True)
            ensemble_predicted_Cl.append(nn_pred_Cl)
            ensemble_residual_Cl.append(nn_res_Cl)
            
        # save all ensemble-computed angular power spectra
        np.save(outdir + 'ensemble_predicted_Cls', np.array(ensemble_predicted_Cl))
        np.save(outdir + 'ensemble_residual_Cls', np.array(ensemble_residual_Cl))

        # compute power spectra for PCA method
        _, pca6_pred_Cl, pca6_res_Cl = angularPowerSpec(cosmo, pca6, bin_min=bin_min, bin_max=bin_max, rearr=info_path+rearr_file, nu_arr=info_path+'/nuTable.txt', 
                                                                    NU_AVG=NU_AVG, N_NU=N_NU, out_dir=outdir + 'angular/', name='pca6', save_spec=True)

        _, noise_Cl, noise_res_Cl = angularPowerSpec(cosmo, noise, bin_min=bin_min, bin_max=bin_max, rearr=info_path+rearr_file,                                                                        nu_arr=info_path+'nuTable.txt',NU_AVG=NU_AVG, N_NU=N_NU, out_dir=outdir + 'angular/', name='noise', save_spec=True)



        # next compute radial power spectra

        cosmo_pka = radialPka(cosmo, n_nu=N_NU)

        noise_pka = radialPka(noise, n_nu=N_NU)
        noise_res_pka = radialPka(noise - cosmo, n_nu=N_NU)

        nn_pka = [np.array(radialPka(m, n_nu=N_NU)) for m in nn_preds]
        nn_res_pka = [np.array(radialPka(m - cosmo, n_nu=N_NU)) for m in nn_preds]

        pca6_pka = radialPka(pca6, n_nu=N_NU)
        pca6_res_pka = radialPka(pca6 - cosmo, n_nu=N_NU)

        # save all pka spectra
        np.save(outdir + 'cosmo_pka', np.array(cosmo_pka))

        np.save(outdir + 'noise_pka', np.array(noise_pka))
        np.save(outdir + 'noise_res_pka', np.array(noise_res_pka))

        np.save(outdir + 'nn_pka', np.array(nn_pka))
        np.save(outdir + 'nn_res_pka', np.array(nn_res_pka))

        np.save(outdir + 'pca6_pka', np.array(pca6_pka))
        np.save(outdir + 'pca6_res_pka', np.array(pca6_res_pka))

