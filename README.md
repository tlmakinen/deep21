# deep21

 [![arXiv](https://img.shields.io/badge/arXiv-2010.15843-b31b1b.svg)](https://arxiv.org/abs/2010.15843) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wQnmelM33Qjq-nHeVD9JkTHXER1PAJM0?hl=en#scrollTo=AL9qQvzFPXcTg)


Repository for deep convolutional neural networks (CNN) to separate cosmological signal from high foreground noise contamination for 21-centimeter large-scale structure observations in the radio spectrum.

![panel-gif](https://raw.githubusercontent.com/tlmakinen/deep21/master/tutorial/panels-white.gif)

Read the full publication here: [https://arxiv.org/abs/2010.15843](https://arxiv.org/abs/2010.15843)

Browser-based tutorial available via this [Google Colab notebook](https://colab.research.google.com/drive/1wQnmelM33Qjq-nHeVD9JkTHXER1PAJM0?hl=en#scrollTo=AL9qQvzFPXcT)

![unet-diagram](https://raw.githubusercontent.com/tlmakinen/deep21/master/tutorial/unet-diagram.png)

Contents:
- `pca_processing`: 
	- `HEALPix` simulation data processing from `.fits` to `.npy` voxel format.
	- Cosmological and foreground simulations generated using the [CRIME package](http://intensitymapping.physics.ox.ac.uk/CRIME.html)
        - Principal Component Analysis Python script `pca_format.py` according to [Alonso et al (2014)](https://arxiv.org/abs/1409.8667)
	- Ideally `pca_script.py` should be run in parallel (each single-sky simulation takes about 3 minutes to process on a standard CPU node)

- `UNet` CNNs implemented in Keras:
    - input and output tensor size: ![(64,64,64,1) \sim (N_x, N_y, N_\nu,$](https://latex.codecogs.com/svg.latex?%2864%2C64%2C64%2C1%29%20%5Csim%20%28N_x%2C%20N_y%2C%20N_%5Cnu%2C) `num_bricks`) for 3D convolutions, ![$(64,64,64) \sim (N_x, N_y, N_\nu)$](https://latex.codecogs.com/svg.latex?%2864%2C64%2C64%29%20%5Csim%20%28N_x%2C%20N_y%2C%20N_%5Cnu%29) for 2D convolutions. 
    - 3D and 2D convolutional model parts stored in respective `unet/unet_Nd.py` files
- `configs`:
   - `.json` parent configuration file with cleaning method and analysis parameters to be edited for user's directory
        
- `data_utils`: 
   - Data loaded using `dataloaders.py` to generate noisy simulations in batch-sized chunks for network to train
   - `my_callbacks.py` for varying learning rate and computing custom metrics during training
- `sim_info`: 
   - frequency (`nuTable.txt`) and HEALPix window (`rearr_nsideN.npy`) indices for `CRIME` simulations
- `train.py`: script for training UNet model. Modify Python dictionary input for appropriate number of training epochs

- `run.sh`:
   - sample slurm-based shell script for training ensemble of models in parallel

- `hyperopt`: 
   - folder for hyperparameter tuning on given dataset

Training Data Availability:

All 100 full-sky simulations used for this analysis are now publicly available [on Globus](https://app.globus.org/file-manager?origin_id=cce6012c-14c2-11ec-90b8-41052087bc27&origin_path=%2F) under the folder `ska2`. Polarised foregrounds and another set of data are available under `ska_polarized` and `ska_sims` respectively. 

The training data used in the published UNet is located under the folder `ska`. Each of the independently-seeded 100 simulations is located under a numbered folder. For instance, for simulation 42's data is structured as:
```
|`sim_42`
|----`cosmo`
|--------`cosmo_i.fits`
|---`fg`
|--------`fg_i.fits`

```
where `i` indexes frequencies from 350 to 691 MHz. To feed the data into `pca_script.py`, the `configs/config.json` file should be modified to point to `ska2`.
	

