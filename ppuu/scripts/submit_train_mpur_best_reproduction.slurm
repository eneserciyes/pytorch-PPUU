#!/bin/bash
#
#SBATCH --job-name=train_MPUR_best_policy
#SBATCH --output=logs/train_MPUR_best_policy_%j.out
#SBATCH --error=logs/train_MPUR_best_policy_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --account="vita"
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
python3 train_MPUR.py \
    -policy 'policy-gauss' \
    -learned_cost

