	# normalize cosmo signal and create validation sets

import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load

if __name__ == '__main__':
	# pull in data for computing mse weighting constant
	data = np.load("/mnt/home/tmakinen/ceph/ska_sims/observed_nsim100.npy")  # observed data
	cosmo_data = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nsim100.npy")
	fg_data = np.load("/mnt/home/tmakinen/ceph/ska_sims/fg_nsim100.npy")

	num_pix = data.shape[0]

	# one sky: 192 tiles
	sky_size = 192
	num_skies_test = 2
	test_indx = num_skies_test * sky_size
	# hide away the test sets first
	np.save("/mnt/home/tmakinen/ceph/ska_sims/obs_test.npy", data[-test_indx:])
	np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_test.npy", cosmo_data[-test_indx:])
	np.save("/mnt/home/tmakinen/ceph/ska_sims/fg_test.npy", fg_data[-test_indx:])

	data = data[:-test_indx]
	cosmo_data = cosmo_data[:-test_indx]
	fg_data = fg_data[:-test_indx]

	# save first dimension length
	num_pix = data.shape[0]


	# FIT SCALERS BUT DO NOT RESCALE DATA

	#cosmo_scaler = StandardScaler()
	#cosmo_scaler.fit(cosmo_data.reshape(-1, 1)) # find standard scaling for each of the 30 freq bands

	# repeat for observed signal
	obs_scaler = StandardScaler()
	obs_scaler.fit(data.reshape(-1, 1))


	# normalize the foreground signal
	fg_scaler = StandardScaler()
	fg_scaler.fit(fg_data.reshape(-1, 1))

	# save scalers
	dump(obs_scaler, './models_network2/obs_scaler.bin', compress=True)
	dump(fg_scaler, './models_network2/fg_scaler.bin', compress=True)
	dump(cosmo_scaler, './models_network2/data_scaler.bin', compress=True)



		# take only slice of input
	sky_size = 192
	train_indx = 78 * sky_size
	val_indx = 20 * sky_size

	data_train = data[:train_indx]
	data_val = data[-val_indx:]

	cosmo_train = cosmo_data[:train_indx]
	cosmo_val = cosmo_data[-val_indx:]

	fg_train = fg_data[:train_indx]
	fg_val = fg_data[-val_indx:]



	# Save training data
	np.save("/mnt/home/tmakinen/ceph/ska_sims/obs_train.npy", data_train)
	np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_train.npy", cosmo_train)
	np.save("/mnt/home/tmakinen/ceph/ska_sims/fg_train.npy", fg_train)

	# save validation data
	np.save("/mnt/home/tmakinen/ceph/ska_sims/obs_val.npy", data_val)
	np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_val.npy", cosmo_val)
	np.save("/mnt/home/tmakinen/ceph/ska_sims/fg_val.npy", fg_val)