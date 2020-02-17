# simple code for reading in the separated "signal" and "foreground"
# measurements, adding them together and saving them for the U-NET 
# to read in easily
# by Lachlan Lancaster

import numpy as np
import healpy as hp
import os

if __name__ == '__main__':
	## NUMBER OF SIMULATIONS TO PROCESS, ANYWHERE BETWEEN 1 AND 100
	NUM_SIMS = 1
	## NSIDE OF THE WINDOWS WE WANT TO CUT OUT, IN PRINCIPLE COULD CHANGE THIS
	WINDOW_NSIDE = 4
	## "global" string with namee to disk directory where simulations are stored
	#dirstr = "/tigress/tmakinen/ska_sims"
	dirstr = "/mnt/home/tmakinen/ceph/ska_sims"



	for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):		
		for SNUM in np.arange(1,NUM_SIMS + 1):
			foreground = np.load("%s/run_fg_s1001/win%03d_fg.npy"%(dirstr,PIX_SELEC))
			cosmo = np.load("%s/run_pkEH_s1%03d/win%03d_cosmo.npy"%(dirstr,SNUM,PIX_SELEC))
			observed = foreground + cosmo

			## **NOTE** NEW OBSERVED SIGNALS ARE SAVED IN DIRECTORIES THAT SHOULD HAVE 
			## TO BE CREATED! YOU CAN EITHER CREATE THEM OR CHANGE HOW THEY'RE SAVED
			
			# create output directory
			if not os.path.exists("%s/obs_s1%03d"%(dirstr,SNUM)): 
				os.mkdir("%s/obs_s1%03d"%(dirstr,SNUM))

			np.save("%s/obs_s1%03d/win_fg1_%03d"%(dirstr,SNUM,PIX_SELEC),observed)

