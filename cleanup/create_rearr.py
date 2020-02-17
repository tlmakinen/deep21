# for creating the rearr.npy array that is used 
# to turn the tiled sky back in to the format it
# is observed in, or read in from the simulations
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
	# gets the indices to rearrange everything
	# "GLOBAL" parameters
	MAP_NSIDE = 256
	WINDOW_NSIDE = 4
	# resolution of the outgoing window
	NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
	# actual side length of window
	WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
	rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))


	# get the array indies in the RING formulation
	inds = np.arange(hp.nside2npix(MAP_NSIDE))
	# transfer these to what they would be in the NESTED formulation
	inds_nest = hp.ring2nest(MAP_NSIDE,inds)

	out = []
	for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
	# get the indices of the pxixels which actually are in the larger pixel
		inds_in = np.where((inds_nest/NPIX_WINDOW)==PIX_SELEC)
		to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
		to_rearr = inds[inds_in]
		to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
		to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH))

		out.append(to_rearr)
	out = np.array(out,dtype=int).flatten()
	#np.save("rearr_nside%d"%(WINDOW_NSIDE),out)
