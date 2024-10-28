import pandas as pd
import pickle as pkl
from rdkit import Chem

#
# A function which gets the smile strings of all of the molecules in the test dataset
#

def get_smiles(dataset_path):
    smiles_df = pd.read_csv(dataset_path)
    return smiles_df["smile_string"]

def find_keys_by_values(dictionary, values):
    return {val: [k for k, v in dictionary.items() if v == val] for val in values}

#
# A function which gets the qm9 indeces for each of the smile strings
#

def find_indeces(smiles, path):
    #reading the dictionary of indeces and smile strings
    #this dictionary has indexes as keys and with smile strings for values
    with open(path, 'rb') as file:
        index_smile_dict = pkl.load(file)

    #inverts the dictionary (so we can search for the qm9 index using a smile string)
    smile_index_dict = {value: key for key, value in index_smile_dict.items()}

    #casts the smile string series into a list 
    smiles_list = smiles.tolist()

    #creates a new ditionary of canoncial smile strings
    canonical_smile_index_dict = {}
    for item in smile_index_dict.items():
        try:
            canonical_string = Chem.CanonSmiles(item[0])
            canonical_smile_index_dict[canonical_string] = item[1]
        except:
            print(f"{item[0]} failed")
    

    #searches for a canonical smile string 
    matching_indeces = []
    for smile_string in smiles_list:
        #convers the smile string to a canonical format
        canonical_smile = Chem.CanonSmiles(smile_string)

        #checks to see if the smile string is in the dictionary
        if canonical_smile in canonical_smile_index_dict:
            #gets the index and appends to the matching indeces list
            matching_indeces.append(canonical_smile_index_dict[Chem.CanonSmiles(smile_string)])
    return matching_indeces

#
# Main Function
#

def main():
    #gets the smile strings
    path_to_csv = "../../data/datasets/qm9/picked_peaks/picked_peaks.csv"
    smiles = get_smiles(path_to_csv)

    #finds the qm9 indeces for each of the smile strings
    index_smile_path = "../../data/metadata/index_smile_dict.pkl"
    matching_qm9_indeces = find_indeces(smiles, index_smile_path)

    print(f"The indeces which overlap are {matching_qm9_indeces}")

    #saves the list as a pkl file
    with open('../../data/metadata/excluded_qm9_indeces.pkl', 'wb') as f:
        pkl.dump(matching_qm9_indeces, f) 
main()
