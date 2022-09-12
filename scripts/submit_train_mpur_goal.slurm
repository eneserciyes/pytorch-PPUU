#!/bin/bash
#
#SBATCH --job-name=train_MPUR_goal
#SBATCH --output=logs/train_MPUR_goal_%j.out
#SBATCH --error=logs/train_MPUR_goal_%j.err
#SBATCH --time=32:00:00
#SBATCH --gres gpu:1
#SBATCH --account="vita"
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
python3 train_MPUR.py \
    -name $name
    -lambda_l $lambda_l \
    -lambda_p $lambda_p \
    -lamdba_g $lambda_g \
    -goal_distance $goal_distance \
    -goal_rollout_len $goal_rollout_len \
    -policy $policy \
    -learned_cost

