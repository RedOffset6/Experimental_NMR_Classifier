import os
import numpy as np
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import sys
import torch

# trait = sys.argv[1]
image_type = sys.argv[1]
trait_list = [sys.argv[2]]
#trait_list = ["alcohol", "aldehyde", "epoxide", "ether", "imine", "alkene", "ketone", "amide"]
# trait_list = ["alkene"]


############################################################################
#                                                                          #
#             DEFINES A FUNCTION WHICH MAKES DATA LOADERS AND THEN         #
#                                  CREATES THEM                            #
#                                                                          #
############################################################################
batch_size = 10
# Function to get subset data loaders
def get_data_loader(fraction, previous_loader, batch_size=batch_size):
    # Get indices from the previous loader

    previous_indices = list(iter(previous_loader.sampler))
   
    # Calculate the size of subset data
    subset_size = int(len(previous_indices) * fraction)

    # Create a sampler for the subset
    subset_sampler = SubsetRandomSampler(np.random.choice(previous_indices, subset_size, replace=False))
   
    # Define data loader for the subset
    loader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler, shuffle=False)
    return loader

############################################################################
#                                                                          #
#               CREATES A CUSTOM IMAGE FOLDER CLASS WHICH EXCLUDES         #
#                             CERTAIN FILE NAMES                           #
#                                                                          #
############################################################################

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

#getting the list of excluded indeces
with open('../../data/metadata/excluded_qm9_indeces.pkl', 'rb') as file:
    excluded_indeces = pickle.load(file)

#uses the excluded indeces to create a list if filenames which have been excluded
excluded_filenames = []
for index in excluded_indeces:
    filename = f"{index}.png"
    excluded_filenames.append(filename)


#loops through the traits
for trait in trait_list:


    dataset_path = f"../../../../msci/project/image_classifier/machine_learning/sorted_image_sets/{image_type}/{trait}_training_data"

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

    # Load the dataset.0
    dataset = CustomImageFolder(root=dataset_path, transform=data_transforms, exclude_filenames=excluded_filenames)
    # Total indices of the dataset
    total_indices = set(range(len(dataset)))

    test_size = int(0.2 * len(dataset))

    # Indices for excluding data for testing
    test_indices = set(np.random.choice(len(dataset), test_size, replace=False))

    # Indices for training data
    train_indices = total_indices - test_indices

    # Convert the set back to a list and then to a numpy array for consistency
    train_indices = np.array(list(train_indices))
    test_indices = np.array(list(test_indices))


    print(f"The length of the test dataset is {test_size}")
    print(f"the length of the dataset is {len(train_indices)}")



    # Create a sampler for the subset
    initial_sampler = SubsetRandomSampler(train_indices)

    print(len(initial_sampler))

    # Define data loader for the subset
    loader = DataLoader(dataset, batch_size=batch_size, sampler=initial_sampler,shuffle=False)

    print(f"the length of the loader is {len(loader)}")

    # Example usage:
    fractions = [0.7, 0.5, 0.25, 0.1, 0.01]  # Fractions of the previous loader's data for each subset
    data_loaders = {}  # Dictionary to store data loaders
    data_loaders[1.0]  = loader  # Use the loader with fraction 1.0 as the previous loader


    for fraction in fractions:
        #if there were originally 1000 training datapoints and the previous loader contains 500 datapoints and we want 0.2 of the original dataset
        #we need to take 0.4 of the 500 datapoints
        #this is done by the two lines below
        number_of_indeces_needed = fraction * len(train_indices)
        fraction_of_previous_dataset = number_of_indeces_needed / (len(loader)*batch_size)
        # Get the data loader for the current fraction

        loader = get_data_loader(fraction_of_previous_dataset, loader)

        
        # Store the loader in the dictionary
        data_loaders[fraction] = loader
        print (f"The length of the data loader is : {len(loader)}")


    #############################################
    #                                           #
    #  ASK CALVIN WHY THIS DOESNT WORK ?!?!?!   #
    #                                           #
    #############################################
        

    # # Get the test loader
    # test_loader = DataLoader(dataset, batch_size=10, sampler=Subset(dataset, test_indices), shuffle=False)

    # Get the test loader
    test_loader = DataLoader(dataset, batch_size=10, sampler=SubsetRandomSampler(test_indices), shuffle=False)

    print(f"The length of the test loader is {len(test_loader)}")

    ############################################################################
    #                                                                          #
    #                          SAVING THE DATA LOADERS                         #
    #                                                                          #
    ############################################################################
    if not os.path.exists(f"../../data/train_loaders/{image_type}/train_loaders"):
        os.makedirs(f"../../data/train_loaders/{image_type}/train_loaders")
    if not os.path.exists(f"../../data/train_loaders/{image_type}/test_loaders"):
        os.makedirs(f"../../data/train_loaders/{image_type}/test_loaders")
    # Save data loaders to a pickle file
    with open(f'../../data/train_loaders/{image_type}/train_loaders/{trait}_train_loader.pkl', 'wb') as f:
        pickle.dump(data_loaders, f)

    # # Save test loader to a pickle file
    # with open(f'../../train_loaders/{image_type}/test_loaders/{trait}_test_loader.pkl', 'wb') as f:
    #     pickle.dump(test_loader, f)

    print("Hello")



#######################################################################################
#                                                                                     #
#    THIS WILL HOPEFULLY ALLOW ME TO EXCLUDE THE FILENAME LIST FROM THE DATALOADER    #
#                                                                                     #
#######################################################################################



# # List of filenames to exclude
# exclude_filenames = ["file1.jpg", "file2.jpg"]

# Custom dataset class inheriting from ImageFolder
# class CustomImageFolder(datasets.ImageFolder):
#     def __init__(self, root, transform=None, exclude_filenames=None):
#         super().__init__(root, transform=transform)
#         # Get full paths of excluded files
#         self.exclude_paths = set(os.path.join(root, cls, fname)
#                                  for cls, _, fnames in self.class_to_idx.items()
#                                  for fname in exclude_filenames)
#         # Filter out excluded files from the samples list
#         self.samples = [(path, label) for path, label in self.samples if path not in self.exclude_paths]

# # Initialize the dataset with the custom class
# dataset_path = 'your_dataset_path'
# dataset = CustomImageFolder(root=dataset_path, transform=data_transforms, exclude_filenames=exclude_filenames)

# # Create the DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)