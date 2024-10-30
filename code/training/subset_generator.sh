#!/bin/bash

#SBATCH --job-name=BEN_IMAGE_GENERATION
#SBATCH --account=phys030544
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
echo 'creating the training files'

#phys phys030544
#chem CHEM014742

#hostname

module load languages/miniconda
echo "before activation: $(which python)"

source activate classifier_env

echo "after activation $(which python)"

python3 subset_generator.py COMBINED





