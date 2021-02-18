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

import json


# OPEN CONFIG FILE
config_file_path = sys.argv[2] + 'configs_deep21.json'

with open(config_file_path) as f:
        configs = json.load(f)

dir_configs = configs["directory_configs"]
pca_configs = configs["pca_params"]
unet_configs = configs["unet_params"]
analysis_configs = configs["analysis_params"]

###############################################################################

###############################################################################
# Define all input and output dirs
model_path = unet_configs['model_path']
data_path = dir_configs['data_path']
info_path = dir_configs['info_path']

# inputs for data and models to test
model_path = '/mnt/home/tmakinen/jobs2/nu_avg/results_logp_1_161/'
data_path = '/mnt/home/tmakinen/ceph/pca_ska/avg/amp_test/'
info_path = '/mnt/home/tmakinen/repositories/deep21/sim_info/'


# make outdirs for each 
plus_names = ['./fg_plus_' + str(i+1) + '/' for i in range(5)]
minus_names = ['./fg_minus_' + str(i+1) + '/' for i in range(5)]
names = [plus_names, minus_names]


outdirs = plus_names + minus_names + ['./fg_control/']
types = ['_plus', '_minus', 'control']


if __name__ == '__main__':

    for l,fg_type in enumerate(['plus', 'minus']):

        fg_name = names[l]

        # compute all the plus amplitude simulations
        o = int(sys.argv[1]) - 1 # slurm indexes +1
        outdir = fg_name[o]

        print('working on simulation %d, writing to %s'%(o+1, outdir))

        pca3 = np.load(data_path + 'pca3_%s_sim%03d.npy'%(fg_type, o+1))
        pca6 = np.load(data_path + 'pca6_%s_sim%03d.npy'%(fg_type, o+1))
        cosmo = np.load(data_path + 'cosmo_%s_sim%03d.npy'%(fg_type, o+1))
        noise = np.load(data_path + 'cosmo_noisy_%s_sim%03d.npy'%(fg_type, o+1))

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
            cosmo_Cl, nn_pred_Cl, nn_res_Cl = angularPowerSpec(cosmo, prediction, bin_min=0, bin_max=160, rearr=info_path + 'rearr_nside8.npy', nu_arr=info_path+'nuTable.txt', 
                                                                    NU_AVG=5, N_NU=32, out_dir=outdir + 'angular/', name='nn', save_spec=True)
            ensemble_predicted_Cl.append(nn_pred_Cl)
            ensemble_residual_Cl.append(nn_res_Cl)
            
        # save all ensemble-computed angular power spectra
        np.save(outdir + 'ensemble_predicted_Cls', np.array(ensemble_predicted_Cl))
        np.save(outdir + 'ensemble_residual_Cls', np.array(ensemble_residual_Cl))

        # compute power spectra for PCA method
        _, pca6_pred_Cl, pca6_res_Cl = angularPowerSpec(cosmo, pca6, bin_min=0, bin_max=160, rearr=info_path+'rearr_nside8.npy', nu_arr=info_path+'/nuTable.txt', 
                                                                    NU_AVG=5, N_NU=32, out_dir=outdir + 'angular/', name='pca6', save_spec=True)

        _, noise_Cl, noise_res_Cl = angularPowerSpec(cosmo, noise, bin_min=0, bin_max=160, rearr=info_path+'rearr_nside8.npy',                                                                        nu_arr=info_path+'nuTable.txt',NU_AVG=5, N_NU=32, out_dir=outdir + 'angular/', name='noise', save_spec=True)



        # next compute radial power spectra

        cosmo_pka = radialPka(cosmo, n_nu=32)

        noise_pka = radialPka(noise, n_nu=32)
        noise_res_pka = radialPka(noise - cosmo, n_nu=32)

        nn_pka = [np.array(radialPka(m, n_nu=32)) for m in nn_preds]
        nn_res_pka = [np.array(radialPka(m - cosmo, n_nu=32)) for m in nn_preds]

        pca6_pka = radialPka(pca6, n_nu=32)
        pca6_res_pka = radialPka(pca6 - cosmo, n_nu=32)

        # save all pka spectra
        np.save(outdir + 'cosmo_pka', np.array(cosmo_pka))

        np.save(outdir + 'noise_pka', np.array(noise_pka))
        np.save(outdir + 'noise_res_pka', np.array(noise_res_pka))

        np.save(outdir + 'nn_pka', np.array(nn_pka))
        np.save(outdir + 'nn_res_pka', np.array(nn_res_pka))

        np.save(outdir + 'pca6_pka', np.array(pca6_pka))
        np.save(outdir + 'pca6_res_pka', np.array(pca6_res_pka))

