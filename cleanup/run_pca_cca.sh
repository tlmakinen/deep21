#!/bin/bash
#SBATCH -p bnl
#SBATCH -N1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-02:30:00
#SBATCH --job-name="unet-3d"
# SBATCH --gres=gpu:v100-32gb:4
#SBATCH --mail-type=ALL --mail-user=tmakinen@princeton.edu
#SBATCH --output pca-clean-%J.log
module load  gcc/7.4.0 cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1 nccl/2.4.2-cuda-10.1 python3/3.7.3

#python3 clean_n2.py
source ~/anaconda3/bin/activate 21cm

# Run sim_format
python3 sim_format_fg.py

python3 sim_format_cosmo.py

# Add foreground to cutouts
python3 add_fg_cosmo.py

# Composite data together
python3 composite_data.py

# Run PCA analysis (3 components)
python pca_format.py

# LATER: run unet analysis
# python remove_foreground_main.py