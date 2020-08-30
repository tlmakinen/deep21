#!/bin/bash

#SBATCH -p cca
#SBATCH --job-name=d21_analysis
#SBATCH --array=1-5
#SBATCH --output=nu_avg_%A_%a.out
#SBATCH --nodes=1
# SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=0-9:00:00
# SBATCH --gres=gpu:v100-32gb:4
#SBATCH --mem=250gb

module load  gcc/7.4.0 cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1 nccl/2.4.2-cuda-10.1 python3/3.7.3
# copy code from repository into jobfile
#cp -r ~/repositories/deep21/* .
#python3 clean_n2.py
source ~/anaconda3/bin/activate tf_gpu
python3 analysis.py $SLURM_ARRAY_TASK_ID

