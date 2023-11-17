import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from utils_public import *

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pdb


class VAE(nn.Module): #Create VAE class inheriting from pytorch nn Module class
    def __init__(self, input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride):
        super(VAE, self).__init__()

        # Create encoder model
        self.encoder = Encoder(input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride)

        #Create decoder after calculating input size for decoder
        decoder_input_size = self.calculate_decoder_input_size(image_size, num_layers, kernel_size, stride)
        self.decoder = Decoder(input_channels, hidden_size, num_layers, latent_dim, decoder_input_size, kernel_size, stride, image_size)

    def calculate_decoder_input_size(self, image_size, num_layers, kernel_size, stride):
        #Function to calculate the input size of the decoder given its architecture
        h, w = image_size
        for _ in range(num_layers):
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
        return h, w

    def reparameterize(self, mu, logvar):
        #Sample from gaussian
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_in):
        #Pass through encoder, reparameterize using mu and logvar as given by the encoder, then pass through decoder
        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

class Encoder(nn.Module): #Encoder model of VAE
    def __init__(self, input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride):
        super(Encoder, self).__init__()

        layers = []
        h, w = image_size
        in_channels = input_channels
        for _ in range(num_layers): # Loop over layers, adding conv2d, layernorm, and relu.
            h = (h - kernel_size) // stride + 1 #Update h and w to compensate for previous layers output
            w = (w - kernel_size) // stride + 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_size, kernel_size, stride),
                    nn.LayerNorm([hidden_size, h, w]),
                    nn.ReLU()
                )
            )
            in_channels = hidden_size #Input channels to later conv layers will just be the hidden size

        self.conv_layers = nn.ModuleList(layers) #Collect convolution layers and layernorm in conv_layers object
        self.final_flatten_size = h * w * hidden_size #Calculate size of final FC output layer
        self.fc_mu = nn.Linear(self.final_flatten_size, latent_dim) #Final FC layer to output mean
        self.fc_logvar = nn.Linear(self.final_flatten_size, latent_dim) #Final FC layer to output logvar

    def forward(self, x): #Forward call for encoder
        for layer in self.conv_layers: #Call conv layers sequentially
            x = layer(x)
        x = x.view(x.size(0), -1) #Flatten x
        mu = self.fc_mu(x) #Get mu and logvar from FC layers
        logvar = self.fc_logvar(x)
        return mu, logvar #Return mu and logvar

class Decoder(nn.Module):  #Decoder model of VAE
    def __init__(self, output_channels, hidden_size, num_layers, latent_dim, decoder_input_size, kernel_size, stride, image_size):
        super(Decoder, self).__init__()
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size

        #Initial fully connected layer
        self.fc = nn.Linear(latent_dim, hidden_size * decoder_input_size[0] * decoder_input_size[1])
        layers = []
        h, w = decoder_input_size
        for layer_num in range(num_layers-1): # Loop over layers, adding conv2dtranspose, layernorm, and relu.
            next_h, next_w = self.calc_size_layer(layer_num, num_layers, kernel_size, stride, image_size) #Calculate the size of the output of the layer
            output_padding_h = next_h - kernel_size - stride*(h-1) #Calculate the output padding to ensure the output size is correct
            output_padding_w = next_w - kernel_size - stride*(w-1)
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size, stride),
                    nn.ReplicationPad2d((0,output_padding_w,0, output_padding_h)),
                    nn.LayerNorm([hidden_size, next_h, next_w]),
                    nn.ReLU()
                )
            )
            h,w = next_h, next_w

        self.deconv_layers = nn.ModuleList(layers) #Collect deconv layers

        next_h, next_w = image_size

        output_padding_h = next_h - kernel_size - stride*(h-1) #Calculate the output padding to ensure the output size is correct
        output_padding_w = next_w - kernel_size - stride*(w-1)
        #Final layer brings the image to the original size
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_size, output_channels, kernel_size, stride),
                                        nn.ReplicationPad2d((0,output_padding_w,0, output_padding_h)))
    def calc_size_layer(self, layer_num, num_layers, kernel_size, stride, image_size):
        #Function to calculate the number of channels for a given layer
        h, w = image_size
        for _ in range(num_layers-layer_num-1):
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
        return h,w


    def forward(self, z):
        z = self.fc(z) #Call initial FC layer
        z = z.view(z.size(0), self.hidden_size, self.decoder_input_size[0], self.decoder_input_size[1])  # Reshape to match the deconvolution input shape
        for layer in self.deconv_layers: #Sequentially call deconv layers
            z = layer(z)
        z = self.final_layer(z)
        return torch.sigmoid(z) #Final sigmoid layer

def loss_function(recon_x, x_out, mu, logvar):
    # VAE loss is a sum of KL Divergence regularizing the latent space and reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x_out, reduction='sum') # Reconstruction loss from Binary Cross Entropy
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL Divergence loss
    return BCE + KLD

def train(epoch, data_in_tensor, data_out_tensor): #Train function for one epoch of training
    model.train()
    train_loss = 0
    num_batches = len(data_in_tensor) // batch_size

    #Tqdm progress bar object contains a list of the batch indices to train over
    progress_bar = tqdm(range(num_batches), desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)

    for batch_idx in progress_bar: #Loop over batch indices
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        data_in = data_in_tensor[start_idx:end_idx] #Gather corresponding data
        data_out = data_out_tensor[start_idx:end_idx] #Gather corresponding data


        optimizer.zero_grad() #Set up optimizer
        recon_batch, mu, logvar = model(data_in) #Call model
        loss = loss_function(recon_batch, data_out, mu, logvar) #Call loss function
        loss.backward() #Get gradients of loss
        train_loss += loss.item() #Append to total loss
        optimizer.step() #Update weights using optimizeer

        # Updating the progress bar
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

    average_train_loss = train_loss / len(data_in_tensor) #Calculate average train loss
    tqdm.write('Epoch: {} \tTraining Loss: {:.3f}'.format(epoch, average_train_loss))

def reconstruct_from_vae(model, masked_topologies, device='cpu'):
    with torch.no_grad():
        data_in = torch.from_numpy(masked_topologies).float()
        data_in = data_in.unsqueeze(1).to(device)
        samples = model(data_in)[0][:,0,:,:].to('cpu').numpy()
        samples = np.round(samples)
    return samples

def plot_reconstruction(originals, masked, reconstructions):
    # Function to plot reconstructed city grids alongside originals
    n = len(originals)
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(9, 2*n))
    for i in range(n): # Loop over the grids
        axes[i,0].imshow(masked[i], cmap = "gray") # Plot masked on the left
        axes[i,1].imshow(reconstructions[i], cmap = "gray") # Plot reconstruction on the left
        axes[i,2].imshow(originals[i], cmap = "gray") #Plot originals on the right
        axes[i,3].imshow(originals[i]-reconstructions[i], cmap = "RdBu", vmin=-1, vmax=1) #Plot error on the right
    fig.tight_layout()
    plt.show()

def evaluate_score(masked_topologies, original_topologies, reconstructed_topologies):
    masks = masked_topologies==0.5 #Identify the masked regions
    correct = reconstructed_topologies==original_topologies #Identify all correctly predicted pixels
    correct_in_mask = np.logical_and(correct, masks) # Identify all correctly predicted pixels within masked regions
    accuracy_fractions = np.sum(correct_in_mask, axis=(1,2))/np.sum(masks, axis=(1,2)) #(correct & mask)/#(mask) for each topology individually
    average_accuracy_fraction = np.mean(accuracy_fractions) #Average of these ratios across test set
    return average_accuracy_fraction

def masking_topo(topologies):
    '''
    topologies = n 64 x 64 images to be masked
    returns masked topologies
    '''
    masked_topologies = []
    for i in trange(len(topologies)):
        mask = random_n_masks(np.array((64,64)), 4, 7).astype(bool)
        topology = topologies[i]
        masked_topology = topology*(1-mask) + 0.5*(mask)
        masked_topologies.append(masked_topology)
    masked_topologies = np.stack(masked_topologies)
    return masked_topologies

# topologies_train.npy is 12000 x 64 x 64. 12000 64x64 images
topologies = np.load("topologies_train.npy")

# constrains_train.npy is 12000 x 5. Contains 2 BC (sparse matrix), 2 loads, vol frac
constraints = np.load("constraints_train.npy", allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu/tpu is available

#masking both constraints and topology
masked_topologies = masking_topo(topologies)
masked_constrains = mask_constraints(constraints)


data_in_tensor = torch.from_numpy(masked_topologies).float().to(device)
data_out_tensor = torch.from_numpy(topologies).float().to(device)

#expand dims of tensor in channel 1
data_in_tensor = data_in_tensor.unsqueeze(1)
data_out_tensor = data_out_tensor.unsqueeze(1)

input_channels = 1
image_size = (64, 64)

latent_dim = 10
hidden_size = 64
num_layers = 3
kernel_size = 3
stride = 2
num_epochs = 20
batch_size = 64

model = VAE(input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
pdb.set_trace()
#Let's look at a model summary
from torchsummary import summary
print((input_channels, image_size[0], image_size[1]))
summary(model, input_size=(input_channels, image_size[0], image_size[1]))


# Main loop
for epoch in range(1, num_epochs + 1):
    train(epoch, data_in_tensor, data_out_tensor)

originals = np.random.choice(np.arange(len(masked_topologies)), size=5, replace=False) #Select 5 random indices
reconstructions = reconstruct_from_vae(model, masked_topologies[originals], device) #Reconstruct
plot_reconstruction(topologies[originals], masked_topologies[originals], reconstructions) #Compare


topologies_test = np.load("topologies_test.npy")
masked_topologies_test = np.load("masked_topologies_test.npy")
reconstructions_test = reconstruct_from_vae(model, masked_topologies_test, device) #Reconstruct
score = evaluate_score(masked_topologies_test, topologies_test, reconstructions_test)
print(f"Final Accuracy: {score:.5f}")