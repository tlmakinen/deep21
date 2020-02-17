# for reading in HEALPix Simulations from Paco and
# reformatting them in to "squares on the sky" for 
# each simualtion, bin-averaging in frequency if
# so desired
# by LTL

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp

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
	## "GLOBAL" parameters
	
	## NUMBER OF FREQUENCY BINS YOU'D LIKE TO LOOK AT
	N_NU = 30 
	## NUMBER OF BINS TO AVERAGE OVER (SET EQUAL TO 1 IF YOU DON'T WANT TO AVERAGE)
	NU_AVG = 1
	assert((N_NU%NU_AVG) ==0)
	## THE HEALPix NSIDE OF THE FUNDAMENTAL SIMULATIONS (SHOULD ALWAYS BE 256)
	MAP_NSIDE = 256
	## NSIDE OF THE WINDOWS WE WANT TO CUT OUT, IN PRINCIPLE COULD CHANGE THIS
	WINDOW_NSIDE = 4
	## NUMBER OF SIMULATIONS TO PROCESS, ANYWHERE BETWEEN 1 AND 100
	NUM_SIMS = 100
	# resolution of the outgoing window
	NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
	# actual side length of window
	WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
	rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))

	print(rearr)

	## "global" string with namee to disk directory where simulations are stored
	## this is also where the new cut-out simulations will be saved
	dirstr = "/tigress/tmakinen/ska_sims"
	

	## THESE ARE THE STRINGS THAT SPECIFY WHETHER YOU ARE BREAKING DOWN THE 
	## FOREGROUND SIMULATIONS OR THE COSMOLOGICAL SIMULATIONS

	## FOR FOREGROUND 	
	#type_str = "fg"
	#type_str2 = "fg"

	## FOR COSMOLOGICAL SIGNAL
	type_str = "pkEH"
	type_str2 = "cosmo"
	

	for SNUM in np.arange(1,NUM_SIMS + 1):
		# Open the Fits files
		fgd = np.array([fits.getdata("%s/run_%s_s1%03d/%s_%03d.fits"%(dirstr,type_str,SNUM,type_str2,nu+1),1) for nu in range(N_NU)],dtype=np.float64)
		# average in frequency bins and transpose
		fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd,NU_AVG)]).T

		# get the array indicies in the RING formulation
		inds = np.arange(hp.nside2npix(MAP_NSIDE))
		# transfer these to what they would be in the NESTED formulation
		inds_nest = hp.ring2nest(MAP_NSIDE,inds)

		# EDIT: since we're not averaging pixels together, no need to rearrange indices
		for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
		# get the indices of the pixels which actually are in the larger pixel
			#print(PIX_SELEC)
			inds_in = np.where((inds_nest/NPIX_WINDOW)==PIX_SELEC)
			#print('inds_in = ', inds_in)
			to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
			#print(to_rearr_inds)
			to_rearr = fgd[inds_in]
			#print(to_rearr)
			to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
			#print(to_rearr)
			to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,NU_AVG))

			np.save("%s/run_%s_s1%03d/win%03d_%s"%(dirstr,type_str,SNUM,PIX_SELEC,type_str2),to_rearr)

