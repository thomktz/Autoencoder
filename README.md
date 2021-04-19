# Autoencoder
Image Autoencoder trained on FFHQ image dataset
Trained from scratch using Pytorch

### Data

The images used to train the model come from the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset)

### The model

Simple Autoencoder `128x128x3 -> latent_space -> 128x128x3`, with `latent_space = 200`

Both Encoder and Decoder contain 3 convolutional layers (`kernel_size = 4, stride = 2`), 3 maxpool layers (`kernel_size = 2, stride = 2`) and *ReLU* activation

It was trained overnight on a laptop GPU, more training should improve the results significantly.

### Results

At `epoch = 200`, ground truth images :

![](https://github.com/thomktz/Autoencoder/blob/main/ground_truth.png)

Reconstructions :

![](https://github.com/thomktz/Autoencoder/blob/main/reconstruction.png)

### Code

Models are created, loaded, saved using a model number. Saving is done automatically every `save_every` epoch, when training ends or when program in interrupted
```python 
except KeyboardInterruption:
  print("Training stopped.")
  save(...)
```
