# define callbacks
import healpy as hp
import numpy as np
import tensorflow as tf
from tensorflow import keras


class transfer(keras.callbacks.Callback):
    
    def __init__(self, val_gen, 
                    num_val, batch_size=48, 
                    patience = 1, is_3d=True,
                    record_spectra = False):
        super().__init__()
        # choose one example at random from validation generator
        start = int(np.random.rand()*num_val)
        self.validation_data = val_gen.datafile[start:(start+768)]
        self.batch_size = batch_size
        self.rearr = np.load("/mnt/home/tmakinen/repositories/21cm-unet/rearr_nside8.npy")
        self.patience = patience
        self.is_3d = is_3d
        self.record_spectra = record_spectra

    
    def on_train_begin(self, logs={}):
        self._data = {
                    'val_avg_transfer': [], 
                    'val_transfer': [],
                    'val_avg_res': [],
                    'val_res': []
                     }
        self.epochs_waited = 0


    def on_epoch_end(self, batch, logs={}):
        
        self.epochs_waited += 1
       
        if self.epochs_waited == self.patience:
            

            
            X_val, y_val = self.validation_data.T[0].T, self.validation_data.T[1].T
            if self.is_3d:
                X_val = np.expand_dims(X_val, axis=-1)
                
            y_predict = self.model.predict(X_val)

            trans = []
            res = []
            for i in range(np.squeeze(y_predict).T.shape[0]):
                y_pred = np.squeeze(y_predict).T[i].T.flatten()  # choose given freq band, then flatten out
                y_pred = y_pred[self.rearr]
                y_pred = hp.map2alm(y_pred)
                y_pred = hp.alm2cl(y_pred)


                # Get Cls for COSMO spectrum
                y_v = np.squeeze(y_val)
                y_v = y_v.T[i].T.flatten()
                y_v = y_v[self.rearr]
                y_v = hp.map2alm(y_v)
                y_v = hp.alm2cl(y_v)

                # compute transfer
                trans.append(np.sqrt((y_pred / y_v))[1:])

                # compute residual power spec
                res.append(np.abs(((y_v - y_pred) / y_v))[1:])

            res_avg = np.mean(np.array(res))
            trans_avg = np.mean(np.array(trans))
            self._data['val_avg_transfer'].append(trans_avg)
            self._data['val_avg_res'].append(res_avg)

            if self.record_spectra:
                self._data['val_transfer'].append(np.array(trans))
                self._data['val_res'].append(np.array(res))

            print(" - val avg transfer: %0.4f"%(trans_avg))
            print(" - val avg res: %0.4f"%(res_avg))
            # reset number of epochs waited
            self.epochs_waited = 0
            
        else:
            # increase num epochs waited
            self.epochs_waited += 1
        return
    
    def get_data(self):
        return self._data