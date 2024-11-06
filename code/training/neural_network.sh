#!/bin/bash

#SBATCH --job-name=BEN_IMAGE_GENERATION
#SBATCH --account=PHYS030544
#SBATCH --partition=gpu_veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
echo 'training the neural network'

#hostname
#previously the queue was gpu_veryshort
#account CHEM014742
module load languages/miniconda
module load libs/cuda/12.0.0-gcc-9.1.0
echo "before activation: $(which python)"

source activate classifier_env

echo "after activation $(which python)"

python3 neural_network.py ether COMBINED ALL






