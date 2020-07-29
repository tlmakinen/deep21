# Script for analysis functions for test data for deep21 predictions
# by TLM


## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import healpy as hp
import h5py
from scipy import fftpack
from scipy.signal import kaiser
import sys,os
###############################################################################

###############################################################################

def weighted_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)


###############################################################################

###############################################################################
# custom loss object for loading model
import tensorflow.keras.backend as K
def custom_loss(y_true, y_pred):
    sig = K.mean(K.std(y_true - y_pred))
    return K.log(sig)  + (keras.metrics.mse(y_true, y_pred) / (2*K.square(sig))) + 10

###############################################################################

###############################################################################
# This routine loads the trained UNet ensemble and makes each ensemble member's
# predictions on the input map

def ensemble_prediction(model_path, num_nets, in_map, outfname):

    if not os.path.exists(outfname):
        os.mkdir(outfname)

    nn_preds = []
    for i in range(num_nets):
        net = keras.models.load_model(model_path + 'best_model_%d.h5'%(i+1), custom_objects={'custom_loss': custom_loss})
        prediction = net.predict(np.expand_dims(in_map, axis=-1), batch_size=48)
        nn_preds.append(prediction)
        del net,prediction

    nn_preds = np.array(nn_preds)
    np.save(outfname + 'nn_preds', np.squeeze(nn_preds))
    
    return np.squeeze(nn_preds)



###############################################################################

###############################################################################
# Define performance metric functions

def compute_logp(y_true, y_pred):
    return np.array([np.mean(((y_true[i] - y_pred[i])**2)/(np.mean(np.std(y_true[i] - y_pred[i])**2)) + np.log(np.std(y_true[i] - y_pred[i]))) for i in range((y_true.shape[0]))])

def compute_mse(y_true, y_pred):
    return np.array([np.mean((y_true[i] - y_pred[i])**2) for i in range(y_true.shape[0])])


###############################################################################

###############################################################################
# This routine computes the angular power spectra statistics for a cleaning 
# method and corresponding true map
def angularPowerSpec(y_true, prediction, bin_min, bin_max, nu_arr, rearr, nu_range=161, nwinds=768, nsims=1, N_NU=32, 
                        NU_AVG=5, out_dir='', name='', save_spec=False):
    
    rearr = np.load(rearr)
    nwinds = 768
    N_NU = N_NU
    NU_START = bin_min
    NU_STOP = N_NU*NU_AVG  
    assert(N_NU == (NU_STOP - NU_START) // NU_AVG)

    #N_SKIP = (N_STOP - N_START) // N_NU
    # get the spetrum of frequenies covered in units of MHz
    (bn,nu_bot,nu_top,z_bot,z_top) = np.loadtxt(nu_arr).T
    nu_arr = ((nu_bot + nu_top)/2.)[:-1]
    nu_arr = nu_arr[NU_START:NU_STOP]#[::N_SKIP]
    nu_arr = np.array([np.mean(i,axis=0) for i in np.split(nu_arr,N_NU)])

    # true map
    cosmo_test = (np.array_split(y_true, y_true.shape[0] / nwinds))

    # cleaned map
    y_pred = (np.array_split(prediction, prediction.shape[0] / nwinds))

    # residual map
    y_res = (np.array_split((prediction - y_true), y_true.shape[0] / nwinds))


    cosmo_Cl = []   # Cls for cosmo spectra
    pred_Cl  = []   # Cls for predicted spectra
    res_Cl   = []   # Cls for residual spectra
  
    for i in range(len(nu_arr)):
        
        
        # Get Cls for COSMO spectrum
        # loops over nsims test set skies
        cos = []
        for cosmo in cosmo_test:
            cosmo0 = (cosmo.T[i].T).flatten()
            cosmo0 = cosmo0[rearr]
            alm_cosmo = hp.map2alm(cosmo0)
            Cl_cosmo = hp.alm2cl(alm_cosmo)
            cos.append(Cl_cosmo)
        
        # save average of Cl over nsims
        cosmo_Cl.append(np.mean(cos, axis=0))


        # Get Cls for the predicted maps
        predicted_cl = []
        for y in y_pred:
            y0 = (y.T[i].T).flatten()
            y0 = y0[rearr]
            alm_y = hp.map2alm(y0); del y0
            Cl_y = hp.alm2cl(alm_y)
            predicted_cl.append(Cl_y); del Cl_y

        # save average of Cl over nsims
        pred_Cl.append(np.mean(predicted_cl, axis=0)); del predicted_cl


        # Get Cls for the residual maps
        residual_cl = []
        for y in y_res:
            y0 = (y.T[i].T).flatten()
            y0 = y0[rearr]
            alm_y = hp.map2alm(y0); del y0
            Cl_y = hp.alm2cl(alm_y)
            residual_cl.append(Cl_y); del Cl_y

        # save average of Cl over nsims
        res_Cl.append(np.mean(residual_cl, axis=0)); del residual_cl

        # save outputs 
        if save_spec:
            
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            np.save(out_dir + name + '_cl_res_nu_%03d'%(nu_arr[i]), np.array(res_Cl[-1]))
            np.save(out_dir + name + '_cl_pred_nu_%03d'%(nu_arr[i]), np.array(pred_Cl[-1]))
            np.save(out_dir + 'cl_cosmo_nu_%03d'%(nu_arr[i]), np.array(cosmo_Cl[-1]))
 
        
    return np.array(cosmo_Cl), np.array(pred_Cl), np.array(res_Cl)
###############################################################################

###############################################################################
# This routine computes the radial power spectra statistics for a cleaning 
# method and corresponding true map



def radialPka(in_map, n_nu=32, num_sims=1, k_min=0.01, k_max=0.2):
    # global params
    MAP_NSIDE = 256
    SIM_NSIDE = MAP_NSIDE
    WINDOW_NSIDE = 8
    NUM_SIMS = 1
    # resolution of the outgoing window
    NPIX_WINDOW = int((MAP_NSIDE/WINDOW_NSIDE)**2)
    # actual side length of window
    WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
    nwinds = int(hp.nside2npix(WINDOW_NSIDE))
    
    # survey volume
    V = (nwinds*WINDOW_LENGTH*WINDOW_LENGTH)
    
    out = []
    for sim in range(num_sims):
        map_s = np.array_split(in_map, len(in_map) // nwinds)[sim]
        
        # window function
        w = kaiser(n_nu, beta=14)
       
        # subtract mean of signal
        map_s = np.array([m - np.mean(m) for m in map_s.T]).T
 
        map_s= np.reshape(map_s, (V, n_nu))
        power_spec = np.sum(np.array([np.abs(fftpack.fft(j*w))**2 for j in map_s]),axis=0)/ V
        
        mid = (len(power_spec) // 2)+1
        out.append(power_spec[1:])  # ignore first mode
    
    k_para = np.linspace(k_min, k_max, len(out[0]))
    
    return k_para, np.squeeze(np.array(out))

