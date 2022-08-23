#!/bin/bash
#
#SBATCH --job-name=eval_MPUR_$name
#SBATCH --output=logs/eval_MPUR_$name_%j.out
#SBATCH --error=logs/eval_MPUR_$name_%j.err
#SBATCH --time=32:00:00
#SBATCH --gres gpu:1
#SBATCH --account="vita"
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

python eval_policy.py \
    -name $name \
    -method policy-MPUR \
    -model_dir models/policy_networks \
    -policy_model $policy \
    -mfile $policy