#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name=generate-trajectories-ppuu
#SBATCH -c 4
#SBATCH --chdir /scratch/izar/erciyes
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.erciyes@epfl.ch
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

echo STARTING AT "$(date)"
module restore dev-env-min

for t in 0 1 2; do python /home/erciyes/Projects/pytorch-PPUU/generate_trajectories.py -map i80 -time_slot $t; done

echo FINISHED at "$(date)"
