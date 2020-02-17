# for reading in HEALPix Simulations from Paco and
# reformatting them in to "squares on the sky" for 
# each simualtion, bin-averaging in frequency if
# so desired
# by LTL

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
import os
import time

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
	N_NU = 64
	GIVE_CONTEXT = False
	SPLIT_FILES = True
	## NUMBER OF AVERAGED BINS OUT (SET EQUAL TO N_NU IF YOU DON'T WANT TO AVERAGE)
	NU_AVG = N_NU
	assert((N_NU%NU_AVG) ==0)
	## NUMBER OF FREQ BINS TO SKIP
	#assert((690%N_NU) == 0)
	N_SKIP = 1 #690 / N_NU
	NU_START = 245 
	## THE HEALPix NSIDE OF THE FUNDAMENTAL SIMULATIONS (SHOULD ALWAYS BE 256)
	MAP_NSIDE = 256
	## NSIDE OF THE WINDOWS WE WANT TO CUT OUT, IN PRINCIPLE COULD CHANGE THIS
	WINDOW_NSIDE = 4
	## NSIDE OF THE CONTEXT WINDOWS THAT ARE REDUCED AND ADDED ON
	CON_NSIDE = WINDOW_NSIDE/2
	TARG_PIX = 1
	## NUMBER OF SIMULATIONS TO PROCESS, ANYWHERE BETWEEN 1 AND 100
	NUM_SIMS = 1
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

	## "global" string with namee to disk directory where simulations are stored
	## this is also where the new cut-out simulations will be saved
	dirstr = "/mnt/home/tmakinen/ceph/ska_sims" # "/tigress/tmakinen/ska_sims"


	## THESE ARE THE STRINGS THAT SPECIFY WHETHER YOU ARE BREAKING DOWN THE 
	## FOREGROUND SIMULATIONS OR THE COSMOLOGICAL SIMULATIONS

	## FOR FOREGROUND 	
	# type_str = "fg"
	# type_str2 = "fg"

	## FOR COSMOLOGICAL SIGNAL
	#type_str = "pkEH"
	#type_str2 = "cosmo"

	## FOR BOTH COSMO AND FOREGROUND
	type_str = ["fg", "pkEH"]
	type_str2 = ["fg", "cosmo"]

	## SPECIFICES OUTPUT
	# output_str = "/tigress/tmakinen/ska_sims"
	output_base = "/mnt/home/tmakinen/ceph/ska_sims/"
	out_type = ['test_data/', "train_data/", "val_data/"]

	if SPLIT_FILES:
		output_str =  [output_base + o for o in out_type]

	t1 = time.time()

	for q in range(len(type_str)):

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
		
		to_save = []
		for SNUM in np.arange(1,NUM_SIMS + 1):

			# load the map
			fgd = np.array([fits.getdata("%s/run_%s_s1%03d/%s_%03d.fits"%(dirstr,type_str[q],1,type_str2[q],nu*N_SKIP+NU_START),1) for nu in range(N_NU)],dtype=np.float64).T
			cosmo = np.array([fits.getdata("%s/run_pkEH_s1%03d/cosmo_%03d.fits"%(dirstr,SNUM,nu*N_SKIP+NU_START),1) for nu in range(N_NU)],dtype=np.float64).T
			fgd = fgd + cosmo
			# average in frequency bins and transpose
			if (NU_AVG!=N_NU):
				fgd = np.array([np.mean(i,axis=0) for i in np.split(fgd.T,NU_AVG)]).T

			for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
				# only perform the rotation to the target pixel if 
				# you are planning on including the context window
				if GIVE_CONTEXT:
					# rotate the selected pixel to the reference "target" pixel
					(lon,lat) = hp.pix2ang(WINDOW_NSIDE,PIX_SELEC,nest=True,lonlat=True)
					r1 = hp.Rotator(rot = (lon,0.,0.))
					r2 = hp.Rotator(rot = (0.,lat,0.))
					# perform actual rotation looping over frequency bands
					fgd_selec = np.array([rlontr.rotate_map_pixel(rlattr.rotate_map_pixel(r2.rotate_map_pixel(r1.rotate_map_pixel(i)))) for i in fgd.T]).T
				else:
					TARG_PIX = PIX_SELEC
					fgd_selec = fgd

				# select pixels which are in the desired window
				inds_in = np.where((inds_nest/NPIX_WINDOW)==TARG_PIX)
				s_in = set(inds_in[0])
				indic_mask = np.array([(i in s_in) for i in inds])
				to_rearr_inds = inds_nest[inds_in] - TARG_PIX*NPIX_WINDOW
				to_rearr = fgd_selec[inds_in]
				to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
				outi = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,NU_AVG))
				#print(outi.shape,  PIX_SELEC)

				# if desired, select the context window around the window
				if GIVE_CONTEXT:
					rotated_map = np.array([rconlon.rotate_map_pixel(rconlat.rotate_map_pixel(rlatt.rotate_map_pixel(rlont.rotate_map_pixel(i)))) for i in fgd_selec.T]).T

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
		
		# save total array
		np.save("%s/%s_nnu%03d_%03dsim"%(output_base,type_str2[q],N_NU,NUM_SIMS),to_save)
		#np.save("%s/fg_nnu%03d_%03dsim"%(output_str,N_NU,NUM_SIMS),to_save)

		if SPLIT_FILES:
			# split into train, test, and validation sets
			test_data = to_save[-2*192:]
			to_save = to_save[:-2*192]
			train_data = to_save[:78*192]
			val_data = to_save[-20*192:]

			arr_list = [test_data, train_data, val_data]
			
			for j in range(len(output_str)):
				
				# make output dirs if not already
				if not os.path.exists(output_str[j]): 
					os.mkdir(output_str[j])

				split_arr = np.array_split(arr_list[j], len(arr_list[j]) / 3)
				for k in range(len(split_arr)):
					np.save(output_str[j] + '%s_%03d'%(type_str2[q],k), split_arr[k].T)

		

			#print(to_save.shape)
		t2 = time.time()
		
		print('total computation time for data arrangement :', t2 - t1)
	


