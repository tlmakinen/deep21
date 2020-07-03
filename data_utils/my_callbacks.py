# define callbacks
import healpy as hp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


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

# from https://github.com/bckenstler/CLR/blob/master/clr_callback.py

class CyclicLR(keras.callbacks.Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())