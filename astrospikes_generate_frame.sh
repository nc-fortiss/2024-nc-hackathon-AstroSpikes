#!/bin/bash
#SBATCH --gres=gpu:1
#singularity exec --nv cuda_11-hpe_cnn.sif poetry install
srun singularity exec --bind /home/lecomte:/home/lecomte --nv cuda_12_astrospikes.sif python3 DataLoading/data_loader.py
