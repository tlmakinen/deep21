# fitting the data with a PCA decomposition 
# of the input data

import numpy as np
from astropy.io import fits
from astropy import units as u
import healpy as hp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os,sys

import time

def log_uniform(r, min_val, max_val):
    point = 0
    if (r <= 0):
        point = -1.0
    else:
        log_min_val = np.log10(min_val)
        log_max_val = np.log10(max_val)
        point = 10.0 ** (log_min_val + r * (log_max_val - log_min_val))
    return point

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


#------USER DEFINED DATA ARRANGEMENT PARAMETERS------#


# INPUTS FROM COMMAND LINE
SNUM = int(sys.argv[3])
dataset_num = int(sys.argv[1])

COMPONENTS = [int(sys.argv[2])]

# OUTPUT DIRECTORIES
dirstr = "/mnt/home/tmakinen/ceph/ska2"
output_base = "/mnt/home/tmakinen/ceph/pca_ska/avg"
out_dir = output_base + '/data_%d/'%(dataset_num) 

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

## what bin we want to start / end from 
NU_START = 1
# number to be processed for deep21
N_NU_OUT = 32
# NUMBER OF FREQS TO SKIP / AVERAGE TOGETHER
NU_AVG = 5 # = N_FREQS // N_FREQ_BINS // N_NU_OUT

# AVERAGE FREQUENCIES ?
DO_NU_AVG = True
# NOISE ADDITION ?
ADD_NOISE = True

ALPHA = None


#---------------------------------------------------------------------

if ALPHA is None:
    r = np.random.rand()
    alpha = log_uniform(r, 0.05, 0.5)
    
else:
    alpha = ALPHA

# total number of frequencies being processed
N_NU = N_NU_OUT * NU_AVG

# index of freqs
NU_STOP = NU_START + N_NU
NU_ARR = np.arange(NU_START, NU_STOP, NU_AVG)

# "GLOBAL" parameters
(NU_L,NU_H) = (1,N_NU_OUT*NU_AVG)
N_SKIP = NU_AVG
N_FREQS = 690
N_FREQ_BINS = 4
assert(((NU_H-NU_L + 1)%NU_AVG) ==0)
MAP_NSIDE = 256
SIM_NSIDE = MAP_NSIDE
WINDOW_NSIDE = 8
NUM_SIMS = 1
# resolution of the outgoing window
NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
# actual side length of window
WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))

# rearrange indices
rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
nwinds = int(hp.nside2npix(WINDOW_NSIDE))


# ACTUAL FREQUENCY MEASUREMENTS
(bn,nu_bot,nu_top,z_bot,z_top) = np.loadtxt("./nuTable.txt").T
nu_arr = ((nu_bot + nu_top)/2.)[:-1]



if __name__ == '__main__':
   
    print('working in frequency range ', nu_arr[NU_START-1], '--', nu_arr[NU_START + (N_NU_OUT*NU_AVG)-2])
 
    # initialize the PCA algorithm
    pca = PCA()
    cosmo_out = np.zeros((int(NUM_SIMS*nwinds),SIM_NSIDE//WINDOW_NSIDE,SIM_NSIDE//WINDOW_NSIDE,N_NU_OUT))
    obs_out = np.zeros((int(NUM_SIMS*nwinds),SIM_NSIDE//WINDOW_NSIDE,SIM_NSIDE//WINDOW_NSIDE,N_NU_OUT))

    x_out = np.zeros((NUM_SIMS*nwinds,SIM_NSIDE//WINDOW_NSIDE,SIM_NSIDE//WINDOW_NSIDE,N_NU_OUT))
    pca_outs = [x_out.copy() for i in range(len(COMPONENTS))]
    outs = [cosmo_out, obs_out]

    out_names = ['cosmo', 'obs']
    
    t1 = time.time()

    for _ in np.arange(1,NUM_SIMS + 1):
        # Open the Fits files for foreground and cosmological signal
        fgd = np.array([fits.getdata("%s/sim_%d/fg/fg_%03d.fits"%(dirstr,SNUM,nu),1) for nu in NU_ARR],dtype=np.float64).T
        cosmo = np.array([fits.getdata("%s/sim_%d/cosmo/cosmo_%03d.fits"%(dirstr,SNUM,nu),1) for nu in NU_ARR],dtype=np.float64).T

        ## ADD NOISE to cosmo shape: (?, NNU)
        if ADD_NOISE:
            mean_nu = [np.mean(nu) for nu in cosmo.T]  # variance of noise is derived from mean at each frequency band
            cosmo_n = np.array([cosmo.T[i] + np.random.normal(loc=0, \
                       scale=alpha*mean_nu[i], size=cosmo.T[i].shape) for i in range(len(cosmo.T))]).T
        

        if DO_NU_AVG:
            # average in frequency bins and transpose
            fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd.T,N_NU_OUT)]).T
            cosmo = np.array([np.mean(i,axis=0) for i in np.split(cosmo.T,N_NU_OUT)]).T
            cosmo_n = np.array([np.mean(i,axis=0) for i in np.split(cosmo_n.T,N_NU_OUT)]).T
            
        else:
            # skip every NU_AVG frequency 
            fgd = fgd.T[::NU_AVG].T
            cosmo = cosmo.T[::NU_AVG].T
            cosmo_n = cosmo_n.T[::NU_AVG].T
            

        obs = fgd + cosmo_n

        # do PCA removal of however many components

        pca.fit(obs)
        obs_pca = pca.transform(obs)
        ind_arr = np.reshape(np.arange(np.prod(obs_pca.shape)),obs_pca.shape)

        
        for s,N_COMP_MASK in enumerate(COMPONENTS):
            mask = np.ones(obs_pca.shape)
            for i in range(N_COMP_MASK,obs_pca.shape[1]):
                mask[ind_arr%obs_pca.shape[1]==i] = 0
            obs_pca1 = obs_pca*mask
            obs_pca_red = pca.inverse_transform(obs_pca1)
            print("Now I'm doing the minimum subtraction...")
            obs_pca_red = obs - obs_pca_red

            # get the array indices in the RING formulation
            inds = np.arange(hp.nside2npix(MAP_NSIDE))
            # transfer these to what they would be in the NESTED formulation  
            inds_nest = hp.ring2nest(MAP_NSIDE,inds)

            sig = obs_pca_red

            for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
                # get the indices of the pxixels which actually are in the larger pixel
                inds_in = np.where((inds_nest//NPIX_WINDOW)==PIX_SELEC)
                to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
                to_rearr = sig[inds_in]
                to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
                to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,N_NU_OUT))
                ind = (_ -1)*nwinds + PIX_SELEC
                pca_outs[s][ind] = to_rearr
                
            
            np.save("%s/pca%d_sim%03d"%(out_dir,N_COMP_MASK,SNUM),pca_outs[s])
             
                    
            
        for s,sig in enumerate([cosmo, obs]):
            for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
                # get the indices of the pxixels which actually are in the larger pixel
                inds_in = np.where((inds_nest//NPIX_WINDOW)==PIX_SELEC)
                to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
                to_rearr = sig[inds_in]
                to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
                to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,N_NU_OUT))
                ind = (_ -1)*nwinds + PIX_SELEC
                outs[s][ind] = to_rearr
                    

            np.save("%s/%s_sim%03d"%(out_dir,out_names[s],SNUM),outs[s])
            
    t2 = time.time()
    print('time for subtraction and arrangement: ', (t2-t1) / 60, ' minutes')
    
    with open(out_dir + '/' + 'params_sim%d.txt'%(SNUM), 'a+') as f:
        f.write('\nnu_start: ' + str(NU_START) + '\n')
        f.write('\nnu_stop: ' + str(NU_STOP) + '\n')
        f.write('\naveraged: ' + str(DO_NU_AVG) + '\n')
        f.write('\nnum freq averaged / skipped : ' + str(NU_AVG) + '\n')
        f.write('\nn_comps removed: ' + str(*COMPONENTS) + '\n')
        f.write('\nnoise alpha : ' + str(alpha))
    f.close()



