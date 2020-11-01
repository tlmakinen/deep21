# PCA processing for CRIME simulation data
## contents:
- `pca_script.py`: main processing script. Inherits directory configurations from main config file
- `make_h5.py`: assembles each `.npy` output from `pca_script.py` into a single h5 file to be read via a dataloader

**NOTE:** these scripts were run 5 different times in the publication analysis (5 noise realixations x 100 simulations) and then shuffled via the dataloader method.

# sample slurm command for disbatch-enabled processing

sbatch -p cca -N 5 -t 0-0:20:0 disBatch -c 3 taskFile
