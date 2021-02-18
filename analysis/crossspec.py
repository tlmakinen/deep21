# Script for running analysis on multiple test data
# by TLM 

## Import the required Libraries
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
import json



###############################################################################

###############################################################################
# Define all input and output dirs

info_path = '/mnt/home/tmakinen/repositories/deep21/sim_info/'

parent_dir = '/mnt/home/tmakinen/ceph/deep21_results/unpolarized/'

rearr_file = info_path +  'rearr_nside4.npy'

# loop through existing directories
directories = ['alpha_minus/', 'alpha_plus/', 
                'beta_minus/', 'beta_plus/']

tests = ['test_%03d/'%(int(i+1)) for i in range(10)]

directories += tests

directories = [parent_dir + d for d in directories]
print('dirs:', directories)
# parallelize over directories
directories = directories[int(sys.argv[1])]

###############################################################################

###############################################################################


# data parameters
N_NU = 64
NU_AVG = 3
WINDOW_NSIDE = 4
N_WINDS = 192

bin_min = 0
bin_max = 192
num_nets = 9
num_sims = 1

#sim_num = 90 + int(sys.argv[1])

# remove mean for radial Pka ?
remove_mean = True

if __name__ == '__main__':
    
    t1 = time.time()
    
    for l,drct in enumerate([directories]):

        # outdir different so that we don't mess up last run
        outdir = drct + 'new_run/'
        print(outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        print('working on %d simulations, writing to %s'%(num_sims, outdir))

        # load maps
        pca3 = np.load(drct + 'pca3.npy')
        pca6 = np.load(drct + 'pca6.npy')
        cosmo = np.load(drct + 'cosmo.npy')
        
        if 'alpha' and 'beta' in outdir:
            noise = np.load(drct + 'noise.npy')
        else:
            noise = np.zeros(cosmo.shape)        

        # make nn prediction
        nn_preds = np.load(drct + 'nn_preds.npy')

        # compute ensemble weights
        w_mse = np.load(drct + 'ensemble_weights.npy')

        # now average maps together according to weights
        ensemble_predicted_map = np.average(nn_preds, weights=w_mse, axis=0)
        
        
        # compute power spectra 
        ensemble_predicted_Cl = []
        ensemble_residual_Cl = []

        # cross power spectra
        ensemble_cross_Cl = []

        for m,prediction in enumerate(nn_preds):
            cosmo_Cl, nn_pred_Cl, nn_res_Cl, nn_cross_Cl = angularPowerSpec(cosmo, prediction, 
                                                                bin_min=bin_min, bin_max=bin_max, 
                                                                rearr=rearr_file, 
                                                                nu_arr=info_path+'nuTable.txt',
                                                                NU_AVG=NU_AVG, N_NU=N_NU, out_dir=outdir + 'angular/', 
                                                                name='nn', save_spec=True)
            ensemble_predicted_Cl.append(nn_pred_Cl)
            ensemble_residual_Cl.append(nn_res_Cl)
            ensemble_cross_Cl.append(nn_cross_Cl)

            
        # save all ensemble-computed angular power spectra
        np.save(outdir + 'ensemble_predicted_Cls', np.array(ensemble_predicted_Cl))
        np.save(outdir + 'ensemble_residual_Cls', np.array(ensemble_residual_Cl))
        np.save(outdir + 'ensemble_cross_Cls', np.array(ensemble_cross_Cl))

        # compute power spectra for PCA method
        _, pca6_pred_Cl, pca6_res_Cl, pca6_cross_Cl = angularPowerSpec(cosmo, pca6, 
                                                        bin_min=bin_min, bin_max=bin_max, 
                                                        rearr=rearr_file, 
                                                        nu_arr=info_path+'/nuTable.txt', 
                                                        NU_AVG=NU_AVG, N_NU=N_NU, 
                                                        out_dir=outdir + 'angular/', 
                                                        name='pca6', save_spec=True)

        _, noise_Cl, noise_res_Cl, noise_cross_Cl = angularPowerSpec(cosmo, noise, 
                                                    bin_min=bin_min, bin_max=bin_max, 
                                                    rearr=rearr_file, 
                                                    nu_arr=info_path+'nuTable.txt',
                                                    NU_AVG=NU_AVG, N_NU=N_NU, 
                                                    out_dir=outdir + 'angular/', 
                                                    name='noise', save_spec=True)



        # next compute radial power spectra

        cosmo_pka = radialPka(cosmo, n_nu=N_NU, 
                                remove_mean=remove_mean)

        noise_pka = radialPka(noise, n_nu=N_NU, 
                                remove_mean=remove_mean)
        noise_res_pka = radialPka(noise - cosmo, n_nu=N_NU, 
                                remove_mean=remove_mean)

        nn_pka = [np.array(radialPka(m, n_nu=N_NU, 
                                remove_mean=remove_mean)) for m in nn_preds]
        nn_res_pka = [np.array(radialPka(m - cosmo, n_nu=N_NU, 
                                remove_mean=remove_mean)) for m in nn_preds]

        pca6_pka = radialPka(pca6, n_nu=N_NU, 
                                remove_mean=remove_mean)
        pca6_res_pka = radialPka(pca6 - cosmo, n_nu=N_NU, 
                                remove_mean=remove_mean)

        # save all pka spectra
        np.save(outdir + 'cosmo_pka', np.array(cosmo_pka))

        np.save(outdir + 'noise_pka', np.array(noise_pka))
        np.save(outdir + 'noise_res_pka', np.array(noise_res_pka))

        np.save(outdir + 'nn_pka', np.array(nn_pka))
        np.save(outdir + 'nn_res_pka', np.array(nn_res_pka))

        np.save(outdir + 'pca6_pka', np.array(pca6_pka))
        np.save(outdir + 'pca6_res_pka', np.array(pca6_res_pka))
   
        # finally, compute cross-spectra

        cosmo_pka = radialPka(cosmo, n_nu=N_NU, 
                                remove_mean=remove_mean)

        noise_cross = radialPka(noise, n_nu=N_NU, 
                                remove_mean=remove_mean, cross_spec=cosmo)

        nn_cross = [np.array(radialPka(m, n_nu=N_NU, 
                                remove_mean=remove_mean, cross_spec=cosmo)) for m in nn_preds]

        pca6_cross = radialPka(pca6, n_nu=N_NU, 
                                remove_mean=remove_mean, cross_spec=cosmo)

        pca3_cross = radialPka(pca3, n_nu=N_NU, 
                                remove_mean=remove_mean, cross_spec=cosmo)

        #save all cross-spectra
     
        np.save(outdir + 'noise_cross', np.array(noise_cross))
        np.save(outdir + 'nn_cross', np.array(nn_cross))
        np.save(outdir + 'pca6_cross', np.array(pca6_pka))
        np.save(outdir + 'pca3_cross', np.array(pca3_pka))





             
    t2 = time.time()

    print('finished computing test stats. \n process took %d minutes, \n output located in %s'%((t2-t1)/60., outdir))
