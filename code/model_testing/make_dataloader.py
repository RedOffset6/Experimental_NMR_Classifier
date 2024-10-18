import os
import numpy as np
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import sys
import torch

############################################################################
#                                                                          #
#             CODE WHICH CREATES DATALOADERS WHICH CONTAIN THE EXPT        #
#                                  TEST DATA                               #
#                                                                          #
############################################################################
batch_size = 10


dataset_path = f"../../data/datasets/qm9/images"

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((196, 196))
])

batch_size = 10

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)

print(f"the length of the dataset is {len(dataset)}")

# Get the test loader
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

print(f"The length of the test loader is {len(test_loader)}")

############################################################################
#                                                                          #
#                          SAVING THE DATA LOADER                          #
#                                                                          #
############################################################################

if not os.path.exists(f"../../data/test_loaders/qm9"):
    os.makedirs(f"../test_loaders/qm9")

# Save test loader to a pickle file
with open(f'../../data/test_loaders/qm9/alkene_hsqc.pkl', 'wb') as f:
    pickle.dump(test_loader, f)