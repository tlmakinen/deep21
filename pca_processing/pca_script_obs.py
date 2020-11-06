# fitting the data with a PCA decomposition 
# of the input data
# by TLM

import numpy as np
from astropy.io import fits
from astropy import units as u
import healpy as hp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os,sys
import json

import time


# ---------------------------------------------------------------------------------
def log_uniform(r, min_val, max_val):
    point = 0
    if (r <= 0):
        point = -1.0
    else:
        log_min_val = np.log10(min_val)
        log_max_val = np.log10(max_val)
        point = 10.0 ** (log_min_val + r * (log_max_val - log_min_val))
    return point


# ---------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------
# for rotating HEALPix maps on the sphere
def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map

#------USER DEFINED DATA ARRANGEMENT PARAMETERS------#

# OPEN CONFIG FILE
config_file_path = sys.argv[3] + 'configs_deep21.json'

with open(config_file_path) as f:
        configs = json.load(f)

dir_configs = configs["directory_configs"]
pca_configs = configs["pca_params"]

# INPUTS FROM COMMAND LINE
SNUM = int(sys.argv[2])
dataset_num = int(sys.argv[1])

COMPONENTS = pca_configs["N_COMP_MASK"]

# OUTPUT DIRECTORIES
dirstr = dir_configs["sim_path"]
output_base = dir_configs["data_path"]
out_dir = output_base + '/data_%d/'%(dataset_num)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

## what bin we want to start / end from 
NU_START = pca_configs["NU_START"]
# number to be processed for deep21
N_NU_OUT = pca_configs["N_NU_OUT"]
# NUMBER OF FREQS TO SKIP / AVERAGE TOGETHER
NU_AVG = pca_configs["NU_AVG"] # = N_FREQS // N_FREQ_BINS // N_NU_OUT

# AVERAGE FREQUENCIES ?
DO_NU_AVG = bool(pca_configs["DO_NU_AVG"])
# NOISE ADDITION ?
ADD_NOISE = bool(pca_configs["ADD_NOISE"])

ALPHA = None


# ---------------------------------------------------------------------------------
if ALPHA is None:
    r = np.random.rand()
    alpha = log_uniform(r, 0.05, 0.5)
    
else:
    alpha = ALPHA

# total number of frequencies being processed
N_NU = N_NU_OUT * NU_AVG

# index of freqs
NU_STOP = NU_START + N_NU
NU_ARR = np.arange(NU_START, NU_STOP)

# "GLOBAL" parameters
(NU_L,NU_H) = (1,N_NU_OUT*NU_AVG)
N_SKIP = NU_AVG
N_FREQS = 690
N_FREQ_BINS = 4
assert(((NU_H-NU_L + 1)%NU_AVG) ==0)
MAP_NSIDE = 256
SIM_NSIDE = MAP_NSIDE
WINDOW_NSIDE = 4
NUM_SIMS = 1
# resolution of the outgoing window
NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
# actual side length of window
WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))

# rearrange indices
rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
nwinds = int(hp.nside2npix(WINDOW_NSIDE))


# ACTUAL FREQUENCY MEASUREMENTS
nutable = dir_configs["info_path"] + "nuTable.txt"
(bn,nu_bot,nu_top,z_bot,z_top) = np.loadtxt(nutable).T
nu_arr = ((nu_bot + nu_top)/2.)[:-1]

# ---------------------------------------------------------------------------------

if __name__ == '__main__':
   
    print('working in frequency range ', nu_arr[NU_START-1], '--', nu_arr[NU_START + (N_NU_OUT*NU_AVG)-2])

    # choose rotations on the sphere:
    # draw theta, phi uniformly on intervals [0, pi]; [0, 2*pi]
    rot_theta = np.random.uniform(low=-1.0, high=1.0)*(np.pi / 2)
    rot_phi   = np.random.uniform(low=-1.0, high=1.0)*2*(np.pi)
    print('rotating maps by theta = ', rot_theta, 'phi = ', rot_phi)    
 
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
        obs = np.array([fits.getdata("%ssim_%d/obs/obs_%03d.fits"%(dirstr,SNUM,nu),1) for nu in NU_ARR],dtype=np.float64).T
        cosmo = np.array([fits.getdata("%ssim_%d/cosmo/cosmo_%03d.fits"%(dirstr,SNUM,nu),1) for nu in NU_ARR],dtype=np.float64).T

        ## ADD NOISE to cosmo shape: (?, NNU)
        if ADD_NOISE:
            mean_nu = [np.mean(nu) for nu in cosmo.T]  # variance of noise is derived from mean at each frequency band
            cosmo_n = np.array([cosmo.T[i] + np.random.normal(loc=0, \
                       scale=alpha*mean_nu[i], size=cosmo.T[i].shape) for i in range(len(cosmo.T))]).T


        if DO_NU_AVG:
            # average in frequency bins and transpose
            obs = np.array([np.mean(i,axis=0) for i in np.split(obs.T,N_NU_OUT)]).T
            cosmo = np.array([np.mean(i,axis=0) for i in np.split(cosmo.T,N_NU_OUT)]).T
            cosmo_n = np.array([np.mean(i,axis=0) for i in np.split(cosmo_n.T,N_NU_OUT)]).T

        else:
            # skip every NU_AVG frequency 
            obs     = obs.T[::NU_AVG].T
            cosmo   = cosmo.T[::NU_AVG].T
            cosmo_n = cosmo_n.T[::NU_AVG].T


        obs = obs + cosmo_n - cosmo # add in just the noise


        # do random rotation of map on sky
        if bool(pca_configs["DO_ROT"]): 
            print("Now I'm rotating each map on the sphere...")
            obs = np.array([rotate_map(o, rot_theta, rot_phi) for o in obs.T]).T
            cosmo = np.array([rotate_map(o, rot_theta, rot_phi) for o in cosmo.T]).T        
        
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
        f.write('\nn_comps removed: ', COMPONENTS, '\n')
        f.write('\nnoise alpha : ' + str(alpha))
    f.close()



