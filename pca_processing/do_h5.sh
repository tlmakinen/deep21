#!/bin/bash
#SBATCH -p cca
#SBATCH --job-name=collect_dat
#SBATCH --array=1-3
#SBATCH --output=collect_%A_%a.out
#SBATCH --nodes=1
#SBATCH --time=0-0:40:00
#SBATCH --mem=50gb

module load  gcc/7.4.0 cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1 nccl/2.4.2-cuda-10.1 python3/3.7.3
# copy code from repository into jobfile
#cp -r ~/repositories/deep21/* .
#python3 clean_n2.py
source ~/anaconda3/bin/activate tf_gpu
python3 make_h5.py $SLURM_ARRAY_TASK_ID /mnt/home/tmakinen/repositories/deep21/configs/
