# deep21
Repository for deep convolutional neural networks (CNN) to separate cosmological signal from high foreground noise contamination for 21-centimeter large-scale structure observations in the radio spectrum.

Read the full publication here: [](https://arxiv.org/abs/2010.15843)

![unet-diagram](https://raw.githubusercontent.com/tlmakinen/deep21/master/tutorial/unet-diagram.png)

Browser-based tutorial available via this [Google Colab notebook](https://colab.research.google.com/drive/1wQnmelM33Qjq-nHeVD9JkTHXER1PAJM0?hl=en#scrollTo=AL9qQvzFPXcT)

Contents:
- `pca_processing`: 
	- `HEALPix` simulation data processing from `.fits` to `.npy` voxel format.
	- Cosmological and foreground simulations generated using the [CRIME package](http://intensitymapping.physics.ox.ac.uk/CRIME.html)
        - Principal Component Analysis Python script `pca_format.py` according to [Alonso et al (2014)](https://arxiv.org/abs/1409.8667)
	- Ideally `pca_script.py` should be run in parallel (3 minutes per simulation)

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

