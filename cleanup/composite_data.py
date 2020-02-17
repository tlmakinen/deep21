## This takes the windows that are cut out and saved using "sim_format.py"
## and put them all together in to a single file that can be moved around 
## by itself or alternatively read in at once for a training sesssion of 
## the U-NET.
# by Lachlan Lancaster

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


if __name__ == '__main__':
	## Number of simulations to process
	NUM_SIMS = 1
	## HEALPix nside of the simulations, should always be 256
	SIM_NSIDE = 256
	## The HELAPix nside of the windows that were cut out
	WINDOW_NSIDE = 4
	## Number of frequency bins that are being read in
	N_NU = int(64)
	#assert((690%N_NU) == 0)
	N_SKIP = 1 #690 / N_NU
	# string that gives the location of the simulations to be processed
	#dirstr = "/tigress/tmakinen/ska_sims"
	dirstr = "/mnt/home/tmakinen/ceph/ska_sims"
	savestr = "/home/tmakinen/datasets"
	nwinds = hp.nside2npix(WINDOW_NSIDE)

	## Arrays to be filled with data and saved
	x_in = np.zeros((int(NUM_SIMS*nwinds),int(SIM_NSIDE/WINDOW_NSIDE),int(SIM_NSIDE/WINDOW_NSIDE),N_NU))
	x_out = np.zeros((int(NUM_SIMS*nwinds),int(SIM_NSIDE/WINDOW_NSIDE),int(SIM_NSIDE/WINDOW_NSIDE),N_NU))
	cosmo_arr = np.zeros((int(NUM_SIMS*nwinds),int(SIM_NSIDE/WINDOW_NSIDE),int(SIM_NSIDE/WINDOW_NSIDE),N_NU))

	## Loop over the pixels that are selected on the sky
	for PIX_SELEC in np.arange(nwinds):
		## Loop over the number of the simulation
		for SNUM in np.arange(1,NUM_SIMS + 1):
			## read in foreground window cut-out
			foreground = np.load("%s/run_fg_s1%03d/win%03d_fg.npy"%(dirstr,SNUM,PIX_SELEC))
			## read in cosmological signal
			cosmo = np.load("%s/run_pkEH_s1%03d/win%03d_cosmo.npy"%(dirstr,SNUM,PIX_SELEC))
			## read in the observed signal, which I guess I decided to process separately
			observed = np.load("%s/obs_s1%03d/win_fg1_%03d.npy"%(dirstr,SNUM,PIX_SELEC))
			## This finds the right place to put them in the output file
			ind = (SNUM-1)*nwinds + PIX_SELEC
			x_in[ind] = observed
			x_out[ind] = foreground
			cosmo_arr[ind] = cosmo
	## Save everything when you're done
	np.save("%s/observed_mid_nnu%d_nsim%d"%(dirstr,N_NU,NUM_SIMS),x_in)
	np.save("%s/fg_mid_nnu%d_nsim%d"%(dirstr,N_NU,NUM_SIMS),x_out)
	np.save("%s/cosmo_mid_nnu%d_nsim%d"%(dirstr,N_NU,NUM_SIMS),cosmo_arr)