# deep21
Repository for deep convolutional neural networks (CNN) to separate cosmological signal from high foreground noise contamination.
Contents:
- `cleanup`: 
	- data processing from `.fits` data file format
        - Principal Component Analysis script according to Alonso et al (2014) https://arxiv.org/abs/1409.8667

- `UNet` CNNs implemented in Keras:
        - input and output tensor size: (32, 32, 32, 3) ~ (x, y, $\nu$, `num_bricks`)
        - 3D and 2D convolutional model parts stored in respective `unet_Nd.py` files
        - `data_generators.py`: method for accessing batch-size chunks of training + validation data

- `train_.py`: scripts for training appropriate model. Modify Python script for appropriate number of training epochs


