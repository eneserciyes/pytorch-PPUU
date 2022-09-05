#!/bin/bash
#
#SBATCH --job-name=train_MPUR_goal_predictor_t_steps
#SBATCH --output=logs/train_MPUR_goal_predictor_t_steps_%j.out
#SBATCH --error=logs/train_MPUR_goal_predictor_t_steps_%j.err
#SBATCH --time=32:00:00
#SBATCH --gres gpu:1
#SBATCH --account="vita"
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate PPUU

cd ../
python3 train_MPUR.py \
    -name goal_predictor_with_goal_cost \
    -model_dir models/policy_networks \
    -mfile MPUR-policy-gauss-name=rg_initial_bug_fixed-model=vae-lambdal=0.2-lambdap=1.0-lambdag=1.0-goaldist=5-goalrollout=30-learnedcost=True-seed=1best.model \
    -lambda_l 0.2 \
    -lambda_p 1.0 \
    -lambda_g 0.0 \
    -goal_distance 5 \
    -goal_rollout_len 30 \
    -policy 'policy-gauss' \
    -learned_cost \
    -train_policy 'goal'
