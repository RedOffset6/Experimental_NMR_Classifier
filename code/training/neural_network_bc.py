####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#this file aims to create a convolutional neural network which can determine between the mock cosy images I have gernerates
#in the mk 1 version I aim to determine between ~20000 molecules. ~10000 alkenes and ~10000 non alkenes
#these alkenes are stored in the following datastructure

# alkene_training_data/
# ├── alkenes/
# │   ├── 1111.png
# │   ├── 2222.png
# │   └── ...
# └── non_alkenes/
#     ├── 3333.png
#     ├── 4444.png
#     └── ...



#importing packages
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
import sys
import pickle as pkl
#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot

#BATCH JOB VERSION
# trait_key = int(sys.argv[1])
# print(f"trait key = {trait_key}")
# trait_list = ["alcohol", "aldehyde", "epoxide", "ether", "imine", "alkene", "ketone", "amide"]
# trait = trait_list[trait_key]

#NON BATCH VERSION
trait = sys.argv[1]

experiment_type = sys.argv[2]
subset_size = sys.argv[3]

############################################################################
#                                                                          #
#       READS A DICTIONARY FROM DISK WHICH STORERES SMILE STRINGS          #
#                    WITH QM9DB INDECES AS THEIR KEYS                      #
#                                                                          #
############################################################################

index_smile_dict = read_pickle("index_smile_dict.pkl")


#Custom dataset class inheriting from ImageFolder
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, exclude_filenames=None):
        super().__init__(root, transform=transform)
        # Get full paths of excluded files
        self.exclude_paths = set(os.path.join(root, cls, fname)
                                 for cls, fnames in self.class_to_idx.items()
                                 for fname in exclude_filenames)
        # Filter out excluded files from the samples list
        self.samples = [(path, label) for path, label in self.samples if path not in self.exclude_paths]


#molecule_plotter(index_smile_dict["1530"])

#positive_list = read_pickle(f"{trait}_list.pkl")

############################################################################
#                                                                          #
#       READS THE IMAGES INTO PYTORCH TENSORS AND SETS THE DEVICE TO       #
#                          BE THE GRAPHICS CARD                            #
#                                                                          #
############################################################################

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 100
batch_size = 10
learning_rate = 0.001
num_training_samples = 1280

#using cude to set the device to be the gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################################




############################################################################
#                                                                          #
#                  READING IN THE TRAINING DATALOADER                      #
#                                                                          #
############################################################################

print("PRINTING THE PATH THAT IM TRYING TO OPEN")
print(f'../../data/train_loaders/{experiment_type}/train_loaders/{trait}_train_loader.pkl')


#load the data loader list from the pickle file:
with open(f'../../data/train_loaders/{experiment_type}/train_loaders/{trait}_train_loader.pkl', 'rb') as f:
    train_loaders = pkl.load(f)

############################################################################
#                                                                          #
#                        SETTING UP A MODEL CLASS                          #
#                                                                          #
############################################################################

#DETERMINING THE SHAPE OF THE DATASET



# # Get the first batch
# dataiter = iter(train_loader)
# images, _ = next(dataiter)

# # Access the shape of the first image
# image_shape = images[0].shape

# print("Shape of the first image:", image_shape)
# #print(f"The size of the subset dataset is {len(subset_dataset)}")
# print("HELLO\n\n\n\n\n")
import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, 5)  # Increase the kernel size for larger input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)  # Increase the kernel size
        self.conv3 = nn.Conv2d(64, 64, 5)  # Increase the kernel size
        # Calculate the linear layer input size based on the dimensions of the output from the last convolutional layer
        self.fc1 = nn.Linear(64 * 42 * 42, 64)  # Adjust the linear layer input size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #print(f"The starting shape of the nn is: {x.shape}")
        x = F.relu(self.conv1(x))  # N, 32, 196, 196
        #print(f"The current shape is : {x.shape}")
        x = self.pool(x)  # N, 32, 98, 98
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.conv2(x))  # N, 64, 94, 94
        #print(f"The current shape is : {x.shape}")
        x = self.pool(x)  # N, 64, 47, 47
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.conv3(x))  # N, 64, 43, 43
        #print(f"The current shape is : {x.shape}")
        x = torch.flatten(x, 1)  # N, 64 * 43 * 43
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.fc1(x))  # N, 64
        #print(f"The current shape is : {x.shape}")
        x = self.fc2(x)  # N, 10
        #print(f"The final shape is : {x.shape}")
        return x

###################################################################################################
def train_and_save_nn(train_loader, fraction):
    num_images = len(train_loader)*10
    # print(f"the length of the train loader is {num_images}")
    #creates the model, criterion, and optimizer
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #creating list to put the running losses in 
    running_losses = []

    n_total_steps = len(train_loader)
    #loops through epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        #iterates through the images
        for i , (images, labels) in enumerate(train_loader):
            #print(f"i = {i}")
            images = images.to(device)
            labels = labels.to(device)
            # forwards pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        epoch_loss = running_loss / n_total_steps
        running_losses.append(epoch_loss)  # Append the epoch loss
        
        print(f"[{epoch+1}] loss: {epoch_loss:.3f}")
        

    print("Finish Training")
    #saves the experiment type followed by theg trait and the number of datapoints used
    #PATH = f"models/{experiment_type}/{trait}/[{fraction}]{trait}{num_images}cnn.pth"
    destination_dir = f"../../data/models/{experiment_type}/{trait}"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    #PATH = f"{destination_dir}/[{subset_size}]{trait}{num_images}cnn.pth"
    # Save both model and running losses
    PATH = f"../../data/models/{experiment_type}/{trait}/[{fraction}]{trait}{num_images}cnn.pth"
    print(f"trying to save the model under the following path {PATH}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'running_losses': running_losses
    }, PATH)

#some logic that mean if ALL is parsed as the subset size then all training dataset sizes will be used but if only one fraction is inputted only this fraction will be used
if subset_size == "ALL":
    for fraction, train_loader in train_loaders.items():
        train_and_save_nn(train_loader, subset_size)
else:
    for fraction, train_loader in train_loaders.items():
        print (f"{fraction}, {len(train_loader)}")
    train_and_save_nn(train_loaders[float(subset_size)], subset_size)

