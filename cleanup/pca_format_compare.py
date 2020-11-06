# for formatting the data with a PCA decomposition 
# of the input data

import numpy as np
from astropy.io import fits
from astropy import units as u
import healpy as hp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os

def gen_rearr(nside):
# recursive funtion for finding the right 
# ordering for the nested pixels 
    if (nside==1):
        return np.array([0,1,2,3])
    else:
        smaller = np.reshape(gen_rearr(nside-1),(2**(nside-1),2**(nside-1)))
        npixsmaller = 2**(2*(nside-1))
        top = np.concatenate((smaller,smaller+npixsmaller),axis=1)
        bot = np.concatenate((smaller+2*npixsmaller,smaller+3*npixsmaller),axis=1)
        whole = np.concatenate((top,bot))
        return whole.flatten()



if __name__ == '__main__':
    # "GLOBAL" parameters
    (NU_L,NU_H) = (1,30)
    DO_NU_AVG = False
    SPLIT_FILES = True
    NU_AVG = 64   # EDIT
    ADD_NOISE = True
    #assert(((NU_H-NU_L + 1)%NU_AVG) ==0)
    MAP_NSIDE = 256
    WINDOW_NSIDE = 4
    NUM_SIMS = 100
    N_COMP_MASK = 3 # number of PCA components to remove
    # resolution of the outgoing window
    NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
    # actual side length of window
    WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))

    ## Number of frequency bins that are being read in
    N_NU = 30
    assert((690%N_NU) == 0)
    N_SKIP = 1#690 / N_NU

    ## Number of pca components to remove for comparison
    N_COMP_ARR = [3, 4, 5, 6, 7, 8]

    rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
    nwinds = hp.nside2npix(WINDOW_NSIDE)
    # "global" string with name to disk directory
    dirstr = "/mnt/home/tmakinen/ceph/ska_sims"
    output_base = "/mnt/home/tmakinen/ceph/ska_sims/"


    # initialize the PCA algorithm
    pca = PCA()

    for N_COMP_MASK in N_COMP_ARR:

        x_out = np.zeros((NUM_SIMS*nwinds,64,64,30))
        for SNUM in np.arange(1,NUM_SIMS + 1):
            # Open the Fits files for foreground and ccosmological signal
            fgd = np.array([fits.getdata("%s/run_fg_s1%03d/fg_%03d.fits"%(dirstr,SNUM,nu*N_SKIP+1),1) for nu in range(N_NU)],dtype=np.float64).T
            cosmo = np.array([fits.getdata("%s/run_pkEH_s1%03d/cosmo_%03d.fits"%(dirstr,SNUM,nu*N_SKIP+1),1) for nu in range(N_NU)],dtype=np.float64).T
            # average in frequency bins and transpose
            #fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd,NU_AVG)]).T
            #cosmo = np.array([np.mean(i,axis=0) for i in np.split(cosmo,NU_AVG)]).T
            # create the observed signal as a sum of the forground and cosmological signal

            ## ADD NOISE to cosmo shape: (?, NNU)
            if ADD_NOISE:
                mean_nu = [np.mean(nu) for nu in cosmo.T]  # variance of noise is derived from mean at each frequency band
                cosmo = np.array([cosmo.T[i] + np.random.normal(loc=0, scale=0.1*mean_nu[i], size=cosmo.T[i].shape) for i in range(len(cosmo.T))]).T

            obs = fgd + cosmo

            pca.fit(obs)

            obs_pca = pca.transform(obs)
            ind_arr = np.reshape(np.arange(np.prod(obs_pca.shape)),obs_pca.shape)
            mask = np.ones(obs_pca.shape)
            for i in range(N_COMP_MASK,obs_pca.shape[1]):
                mask[ind_arr%obs_pca.shape[1]==i] = 0
            obs_pca = obs_pca*mask
            obs_pca_red = pca.inverse_transform(obs_pca)
            print("Now I'm doing the minimum subtraction...")
            obs_pca_red = obs - obs_pca_red

            # get the array indices in the RING formulation
            inds = np.arange(hp.nside2npix(MAP_NSIDE))
            # transfer these to what they would be in the NESTED formulation  
            inds_nest = hp.ring2nest(MAP_NSIDE,inds)
            

            for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
                # get the indices of the pxixels which actually are in the larger pixel
                inds_in = np.where((inds_nest/NPIX_WINDOW)==PIX_SELEC)
                to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
                to_rearr = obs_pca_red[inds_in]
                to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
                to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,N_NU))
                ind = (SNUM-1)*nwinds + PIX_SELEC
                x_out[ind] = to_rearr

        np.save("%s/pca_%dcomp_reduced_first_nnu%d_nsim%d"%(dirstr,N_COMP_MASK,N_NU,NUM_SIMS),x_out)



        if SPLIT_FILES:
            out_type = ['%d_test_data_noise/'%(N_COMP_MASK), "%d_train_data_noise/"%(N_COMP_MASK), "%d_val_data_noise/"%(N_COMP_MASK)]
            output_str =  [output_base + o for o in out_type]
            # split into train, test, and validation sets
            test_data = x_out[-2*192:]
            x_out = x_out[:-2*192]
            train_data = x_out[:78*192]
            val_data = x_out[-20*192:]
            arr_list = [test_data, train_data, val_data]
            
            for j in range(len(output_str)):
                
                # make output dirs if not already
                if not os.path.exists(output_str[j]): 
                    os.mkdir(output_str[j])

                split_arr = np.array_split(arr_list[j], len(arr_list[j]) / 3)
                for k in range(len(split_arr)):
                    np.save(output_str[j] + 'pca_%03d'%(k), split_arr[k].T)
