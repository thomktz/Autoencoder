# %% Imports
import torch.nn as nn
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_treatment import dataloader, testloader, batch_size, image_size

# %% Variables
in_channels = 3
latent_dims = 200
print_every = 200 #in batches
save_every = 15 #in epochs
log_loss_every = 5 #in batches
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.maxp = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=200, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(in_features=200, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxp(x)
        x = F.relu(self.conv2(x))
        x = self.maxp(x)
        x = F.relu(self.conv3(x))
        x = self.maxp(x)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(x.size(0), -1) # (200, 1, 1) -> (200) 
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ups = nn.Upsample(scale_factor = 2)
        self.fc = nn.Linear(in_features=latent_dims, out_features=200)
        self.conv4 = nn.ConvTranspose2d(in_channels=200, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 200, 1, 1) # unflatten batch of feature vectors to a batch of multi-channel feature maps
    
        x = F.relu(self.conv4(x))
        x = self.ups(x)
        x = F.relu(self.conv3(x))
        x = self.ups(x)
        x = F.relu(self.conv2(x))
        x = self.ups(x)
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

# %%
def load_old_model(old_path):  #If shape changes, else run next cell
    autoencoder = Autoencoder()

    old_autoencoder = torch.load(old_path)
    autoencoder.encoder.conv1.weight = nn.Parameter(old_autoencoder["encoder.conv1.weight"])
    autoencoder.encoder.conv1.bias = nn.Parameter(old_autoencoder["encoder.conv1.bias"])

    autoencoder.encoder.conv2.weight = nn.Parameter(old_autoencoder["encoder.conv2.weight"])
    autoencoder.encoder.conv2.bias = nn.Parameter(old_autoencoder["encoder.conv2.bias"])

    autoencoder.encoder.conv3.weight = nn.Parameter(old_autoencoder["encoder.conv3.weight"])
    autoencoder.encoder.conv3.bias = nn.Parameter(old_autoencoder["encoder.conv3.bias"])

    autoencoder.decoder.conv1.weight = nn.Parameter(old_autoencoder["decoder.conv1.weight"])
    autoencoder.decoder.conv1.bias = nn.Parameter(old_autoencoder["decoder.conv1.bias"])

    autoencoder.decoder.conv2.weight = nn.Parameter(old_autoencoder["decoder.conv2.weight"])
    autoencoder.decoder.conv2.bias = nn.Parameter(old_autoencoder["decoder.conv2.bias"])

    autoencoder.decoder.conv3.weight = nn.Parameter(old_autoencoder["decoder.conv3.weight"])
    autoencoder.decoder.conv3.bias = nn.Parameter(old_autoencoder["decoder.conv3.bias"])
# %%
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("models\\bigger_model.pth"))
autoencoder = autoencoder.to(device)

num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)# %%

# %%
learning_rate = 1e-4  #first run @ 1e-3
batch_size = 64
num_epochs = 300
# %%
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
autoencoder.train()

train_loss_avg = []

print('Training ...')
try:
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        if epoch % save_every == 0:
            torch.save(autoencoder.state_dict(), f"models\\bigger_model_epoch{epoch}.pth")
        for image_batch, _ in dataloader:

            
            image_batch = image_batch.to(device)
            
            # autoencoder reconstruction
            image_batch_recon = autoencoder(image_batch)
            
            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            if num_batches % print_every == 0:
                print(f"Batch no {num_batches}, loss : {loss.item()}")
            
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    torch.save(autoencoder.state_dict(), "models\\bigger_model.pth")
except KeyboardInterrupt:
    print("Entrainement arrêté")
    torch.save(autoencoder.state_dict(), "models\\bigger_model.pth")

# %%
fig = plt.figure()
plt.plot(train_loss_avg[:-1])
plt.xlabel('Epochs')
plt.ylabel('Reconstruction error')
plt.show()
# %%
autoencoder.eval()

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):

    with torch.no_grad():

        images = images.to(device)
        images = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

images, _ = iter(testloader).next()

# First visualise the original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

# Reconstruct and visualise the images using the autoencoder
print('Autoencoder reconstruction:')
visualise_output(images, autoencoder)
# %%
