#!/bin/bash
#
#SBATCH --job-name=train_MPUR_goal_predictor_normalised_first
#SBATCH --output=logs/train_MPUR_goal_predictor__normalised_first_%j.out
#SBATCH --error=logs/train_MPUR_goal_predictor__normalised_first_%j.err
#SBATCH --time=32:00:00
#SBATCH --gres gpu:1
#SBATCH --account="vita"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
python3 train_MPUR.py \
    -name normalised_first \
    -model_dir models/policy_networks \
    -mfile MPUR-policy-gauss-name=value_net_goal-model=vae-lambdal=0.2-lambdap=1.0-lambdag=1.0-goaldist=5-goalrollout=30-learnedcost=True-seed=1best.model \
    -lambda_l 0.2 \
    -lambda_p 1.0 \
    -lambda_g 0.0 \
    -lambda_gp 1.0 \
    -goal_distance 5 \
    -goal_rollout_len 30 \
    -policy 'policy-gauss' \
    -learned_cost \
    -train_policy 'goal'
