# Autoencoder
Autoencoder and PCA reduction on FFHQ image dataset

### Data

The images used to train the model come from the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset)

### The model

Simple Autoencoder `128x128x3 -> latent_space -> 128x128x3`, with `latent_space = 200`

### Results

At `epoch = 200`, 

![](https://github.com/thomktz/Autoencoder/blob/main/ground_truth.png)

![](https://github.com/thomktz/Autoencoder/blob/main/reconstruction.png)
