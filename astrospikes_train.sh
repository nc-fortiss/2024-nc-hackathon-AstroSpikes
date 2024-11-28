#!/bin/bash
#SBATCH --gres=gpu:1
#singularity exec --nv cuda_11-hpe_cnn.sif poetry install
#srun singularity exec --bind /home/lecomte:/home/lecomte --nv cuda_12_astrospikes.sif python3 keras_mobilenet_v1.py
srun singularity exec --bind /home/lecomte:/home/lecomte --nv cuda_12_astrospikes.sif python3 mobilenet_v1_akida_implementation.py
