# deep21
Repository for deep convolutional neural networks (CNN) to separate cosmological signal from high foreground noise contamination for 21-centimeter large-scale structure observations in the radio spectrum.
Contents:
- `cleanup`: 
	- data processing from `.fits` data file format
        - Principal Component Analysis script according to Alonso et al (2014) https://arxiv.org/abs/1409.8667

- `UNet` CNNs implemented in Keras:
    - input and output tensor size: (32, 32, 32, 1) ~ (x, y, ![$\nu$](https://render.githubusercontent.com/render/math?math=%24%5Cnu%24), `num_bricks`) for 3D convolutions, (32,32,32) ~ (x,y,![$\nu$](https://render.githubusercontent.com/render/math?math=%24%5Cnu%24)) for 2D convolutions. 
    - 3D and 2D convolutional model parts stored in respective `unet/unet_Nd.py` files
        
-`data_utils`: 
   - Data loaded using `dataloaders.py` to generate noisy simulations in batch-sized chunks for network to train

- `train_nD.py`: scripts for training appropriate model. Modify Python dictionary input for appropriate number of training epochs


