# for formatting the data with a PCA decomposition 
# of the input data

import numpy as np
from astropy.io import fits
from astropy import units as u
import healpy as hp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
	GIVE_CONTEXT = True
	## NUMBER OF AVERAGED BINS OUT (SET EQUAL TO N_NU IF YOU DON'T WANT TO AVERAGE)
	NU_AVG = N_NU
	assert((N_NU%NU_AVG) ==0)
	## THE HEALPix NSIDE OF THE FUNDAMENTAL SIMULATIONS (SHOULD ALWAYS BE 256)
	MAP_NSIDE = 256
	## NSIDE OF THE WINDOWS WE WANT TO CUT OUT, IN PRINCIPLE COULD CHANGE THIS
	WINDOW_NSIDE = 4
	## NSIDE OF THE CONTEXT WINDOWS THAT ARE REDUCED AND ADDED ON
	CON_NSIDE = WINDOW_NSIDE/2
	TARG_PIX = 1
	## NUMBER OF SIMULATIONS TO PROCESS, ANYWHERE BETWEEN 1 AND 100
	NUM_SIMS = 1
	N_COMP_MASK = 3 # number of PCA components to remove
	# resolution of the outgoing window/context window
	NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
	NPIX_CON = (MAP_NSIDE/CON_NSIDE)**2
	# actual side length of window
	WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
	rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
	CON_LENGTH = int(np.sqrt(NPIX_CON))
	rearr_con = gen_rearr(int(np.log2(MAP_NSIDE/CON_NSIDE)))

	# get the array indies in the RING formulation
	inds = np.arange(hp.nside2npix(MAP_NSIDE))
	# transfer these to what they would be in the NESTED formulation
	inds_nest = hp.ring2nest(MAP_NSIDE,inds)



	rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
	nwinds = hp.nside2npix(WINDOW_NSIDE)
	# "global" string with name to disk directory
	dirstr = "/tigress/tmakinen/ska_sims"
	# initialize the PCA algorithm
	pca = PCA()

	if GIVE_CONTEXT:
		(lont,latt) = hp.pix2ang(WINDOW_NSIDE,TARG_PIX,nest=True,lonlat=True)
		rlont = hp.Rotator(rot = (lont,0.,0.))
		rlatt = hp.Rotator(rot = (0.,latt,0.))
		rlattr = hp.Rotator(rot = (0.,-1*latt,0.))
		rlontr = hp.Rotator(rot = (-1*lont,0.,0.))
		CON_SELEC = hp.ang2pix(CON_NSIDE,lont,latt,nest=True,lonlat=True)
		(CON_LON,CON_LAT) = hp.pix2ang(CON_NSIDE,CON_SELEC,nest=True,lonlat=True)
		rconlat = hp.Rotator(rot = (0.,-1*CON_LAT,0))
		rconlon = hp.Rotator(rot = (-1*CON_LON,0.,0))


	
	x_out = np.zeros((NUM_SIMS*nwinds,64,64,30))
	for SNUM in np.arange(1,NUM_SIMS + 1):
		# Open the Fits files for foreground and ccosmological signal
		fgd = np.array([fits.getdata("%s/run_fg_s1%03d/fg_%03d.fits"%(dirstr,SNUM,nu+1),1) for nu in range(N_NU)],dtype=np.float64).T
		cosmo = np.array([fits.getdata("%s/run_pkEH_s1%03d/cosmo_%03d.fits"%(dirstr,SNUM,nu+1),1) for nu in range(N_NU)],dtype=np.float64).T
		# average in frequency bins and transpose
		#fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd,NU_AVG)]).T
		#cosmo = np.array([np.mean(i,axis=0) for i in np.split(cosmo,NU_AVG)]).T
		# create the observed signal as a sum of the forground and cosmological signal
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
		
		to_save = []
		for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
			# only perform the rotation to the target pixel if 
			# you are planning on including the context window
			if GIVE_CONTEXT:
				# rotate the selected pixel to the reference "target" pixel
				(lon,lat) = hp.pix2ang(WINDOW_NSIDE,PIX_SELEC,nest=True,lonlat=True)
				r1 = hp.Rotator(rot = (lon,0.,0.))
				r2 = hp.Rotator(rot = (0.,lat,0.))
				# perform actual rotation looping over frequency bands
				obs_pca_red_selec = np.array([rlontr.rotate_map_pixel(rlattr.rotate_map_pixel(r2.rotate_map_pixel(r1.rotate_map_pixel(i)))) for i in obs_pca_red.T]).T
			else:
				obs_pca_red_selec = obs_pca_red



			# select pixels which are in the desired window
			inds_in = np.where((inds_nest/NPIX_WINDOW)==TARG_PIX)
			s_in = set(inds_in[0])
			indic_mask = np.array([(i in s_in) for i in inds])
			to_rearr_inds = inds_nest[inds_in] - TARG_PIX*NPIX_WINDOW
			to_rearr = obs_pca_red_selec[inds_in]
			to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
			outi = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,NU_AVG))
			print('pca reduced shape: ', outi.shape)


						# if desired, select the context window around the window
			if GIVE_CONTEXT:
				rotated_map = np.array([rconlon.rotate_map_pixel(rconlat.rotate_map_pixel(rlatt.rotate_map_pixel(rlont.rotate_map_pixel(i)))) for i in obs_pca_red_selec.T]).T

				inds_in_con = np.where((inds_nest/NPIX_CON)==CON_SELEC)
				s_in_con = set(inds_in_con[0])
				indic_mask_con = np.array([(i in s_in_con) for i in inds])
				to_rearr_inds_con = inds_nest[inds_in_con] - CON_SELEC*NPIX_CON
				to_rearr_con = rotated_map[inds_in_con]
				to_rearr_con = (to_rearr_con[np.argsort(to_rearr_inds_con)])[rearr_con]
				to_rearr_con = np.reshape(to_rearr_con,(CON_LENGTH,CON_LENGTH,NU_AVG))
				to_rearr_con = np.array([np.mean(i,axis=0) for i in np.split(to_rearr_con,CON_LENGTH/(WINDOW_NSIDE/CON_NSIDE))])
				to_rearr_con = np.transpose(to_rearr_con,(1,0,2))
				to_rearr_con = np.array([np.mean(i,axis=0) for i in np.split(to_rearr_con,CON_LENGTH/(WINDOW_NSIDE/CON_NSIDE))])
				to_rearr_con = np.transpose(to_rearr_con,(1,0,2))
				outi = np.concatenate((outi.T,to_rearr_con.T)).T
			to_save.append(outi)
	to_save = np.array(to_save)
	print('save arr shape: ', to_save.shape)
		#np.save("%s/context_nsim%d"%(dirstr,NUM_SIMS),to_save)

	np.save("%s/pca_context_%dcomp_reduced_nsim%d"%(dirstr,N_COMP_MASK,NUM_SIMS),to_save)
	
