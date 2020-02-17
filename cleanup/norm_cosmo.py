# normalize cosmo signal and create validation sets

import numpy as np 
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    # load the training data

    data = np.load("/mnt/home/tmakinen/ceph/ska_sims/pca_3comp_reduced_nsim100.npy")
    print('data size: ', data.shape)
    signal = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nsim100.npy")

    pca_6 = np.load("/mnt/home/tmakinen/ceph/ska_sims/pca_6comp_reduced_nsim100.npy")



    # one sky: 192 tiles
    sky_size = 192
    num_skies_test = 2
    test_indx = num_skies_test * sky_size
    # hide away the test sets first
    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_test.npy", data[-test_indx:])
    np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_test.npy", signal[-test_indx:])
    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_6_test.npy", pca_6[-test_indx:])

    pca_6 = pca_6[:-test_indx]
    data = data[:-test_indx]
    signal = signal[:-test_indx]
    # save first dimension length
    num_pix = signal.shape[0]
    # now normalize the cosmo signal for inference
    cosmo_scaler = StandardScaler()
    cosmo_scaler.fit(signal.reshape(-1, 1))  # find standard scaling for each of the 30 freq bands


    num_pix = data.shape[0]
    # normalize the input signal
    pca_scaler = StandardScaler()
    pca_scaler.fit(data.reshape(-1, 1))


    # split the data in to training/validation sets
    # one sky: 192 tiles
    sky_size = 192
    num_skies_train = 78
    train_indx = sky_size * num_skies_train

    x_train = data[:train_indx]
    y_train = signal[:train_indx]
    x_val = data[train_indx:]
    y_val = signal[train_indx:]

    pca_6_train = pca_6[:train_indx]
    pca_6_val = pca_6[train_indx:]
    # save datasets
    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_train.npy", x_train)
    np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_train.npy", y_train)
    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_val.npy", x_val)
    np.save("/mnt/home/tmakinen/ceph/ska_sims/cosmo_val.npy", y_val)

    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_6_val.npy", pca_6_val)
    np.save("/mnt/home/tmakinen/ceph/ska_sims/pca_6_train.npy", pca_6_train)

    # save scaler
    from sklearn.externals.joblib import dump, load
    dump(cosmo_scaler, './models_network1/cosmo_std_scaler.bin', compress=True)
    dump(pca_scaler, './models_network1/pca_std_scaler.bin', compress=True)