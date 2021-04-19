# %% Imports
import torch.nn as nn
import torch
import numpy as np
import torchvision
import time
import glob
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

# %% Save model state

def save(epoch, model, optimizer, loss_list, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list': loss_list[:-1],
            }, path)

def load(path):
    checkpoint = torch.load(path, map_location=device)
    autoencoder = Autoencoder().cuda()
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    return epoch, autoencoder, optimizer, loss_list


# %%
learning_rate = 1e-4  #first run @ 1e-3
batch_size = 64
num_epochs = 1000
# %%

def train(model_number, new_epochs_number):
    if len(glob.glob(f"models\\number_{model_number}\\")) == 0:
        print(f"Creating model number {model_number}")
        autoencoder = Autoencoder()
        autoencoder = autoencoder.to(device)
        optimizer = optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        train_loss_avg = []
        old_epoch = 0
    else:
        old_epoch, autoencoder, optimizer, train_loss_avg = load(f"models\\number_{model_number}\\checkpoint.pth")
        print(f"Succesfully loaded model number {model_number}, continuiing at epoch {old_epoch}")
    print('Training ...')
    start = time.time()
    try:
        autoencoder.train()
        for epoch in range(new_epochs_number):
            train_loss_avg.append(0)
            num_batches = 0
            if epoch+old_epoch % save_every == 0:
                save(epoch+old_epoch, autoencoder, optimizer, train_loss_avg, f"models\\number_{model_number}\\checkpoint_{epoch+old_epoch}.pth")
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                image_batch_recon = autoencoder(image_batch)
                loss = F.mse_loss(image_batch_recon, image_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()           
                train_loss_avg[-1] += loss.item()
                num_batches += 1
                if num_batches % print_every == 0:
                    #print(f"Batch no {num_batches}, loss : {loss.item()}")
                    pass
                
            train_loss_avg[-1] /= num_batches
            print('Epoch [%d / %d] average reconstruction error: %f, elapsed time : %d minutes, remaining : %d' % (epoch+1+old_epoch, new_epochs_number+old_epoch, train_loss_avg[-1], int((time.time()-start)/60), int((time.time()-start)/60*(new_epochs_number-epoch)/(epoch+1))))
        save(epoch+old_epoch, autoencoder, optimizer, train_loss_avg, f"models\\number_{model_number}\\checkpoint.pth")
    except KeyboardInterrupt:
        print("Training stopped.")
        save(epoch+old_epoch, autoencoder, optimizer, train_loss_avg, f"models\\number_{model_number}\\checkpoint.pth")

train(0, 100)


# %%
def show_loss(model_number):
    _ , _ , _ , train_loss_avg = load(f"models\\number_{model_number}\\checkpoint.pth")
    fig = plt.figure()
    plt.plot(train_loss_avg[:-1])
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction error')
    plt.show()

show_loss(0)
# %%
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
    
def reconstruct(model_number):

    old_epoch, autoencoder, optimizer, train_loss_avg = load(f"models\\number_{model_number}\\checkpoint.pth")
    autoencoder.eval()
    images, _ = iter(testloader).next()

    # First visualise the original images
    print('Original images')
    show_image(torchvision.utils.make_grid(images[1:50],10,5))
    plt.show()

    # Reconstruct and visualise the images using the autoencoder
    print('Autoencoder reconstruction:')
    visualise_output(images, autoencoder)

reconstruct(0)
# %%
