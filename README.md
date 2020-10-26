# deep21
Repository for deep convolutional neural networks (CNN) to separate cosmological signal from high foreground noise contamination for 21-centimeter large-scale structure observations in the radio spectrum.

Browser-based tutorial available via this [Google Colab notebook](https://colab.research.google.com/drive/1wQnmelM33Qjq-nHeVD9JkTHXER1PAJM0?hl=en#scrollTo=AL9qQvzFPXcT)

Contents:
- `cleanup`: 
	- data processing from `.fits` data file format
        - Principal Component Analysis Python script `pca_format.py` according to Alonso et al (2014) https://arxiv.org/abs/1409.8667

- `UNet` CNNs implemented in Keras:
    - input and output tensor size: (64, 64, 64, 1) ~ (x, y, ![$\nu$](https://render.githubusercontent.com/render/math?math=%24%5Cnu%24), `num_bricks`) for 3D convolutions, (64,64,64) ~ (x,y,![$\nu$](https://render.githubusercontent.com/render/math?math=%24%5Cnu%24)) for 2D convolutions. 
    - 3D and 2D convolutional model parts stored in respective `unet/unet_Nd.py` files
- `configs`:
   - `.json` parent configuration file with cleaning method and analysis parameters
        
- `data_utils`: 
   - Data loaded using `dataloaders.py` to generate noisy simulations in batch-sized chunks for network to train
   - `my_callbacks.py` for computing transfer function accuracy during training
- `sim_info`: 
   - frequency (`nuTable.txt`) and HEALPix window (`rearr_nsideN.npy`) indices for simulations
- `train_nD.py`: scripts for training appropriate model. Modify Python dictionary input for appropriate number of training epochs

- `hyperopt`: 
   - folder for hyperparameter tuning on given dataset

