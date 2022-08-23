import os

EXPERIMENTS = [
    # 1
    {
        "name": "default",
        "lambda_l": 0.2,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 5,
        "goal_rollout_len": 30,
    },
    # 2
    {
        "name": "mid_rollout",
        "lambda_l": 0.2,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 5,
        "goal_rollout_len": 20,
    },
    # 3
    {
        "name": "longer_goal_dist",
        "lambda_l": 0.2,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 10,
        "goal_rollout_len": 30,
    },
    # 4
    {
        "name": "shorter_goal_dist",
        "lambda_l": 0.2,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 2,
        "goal_rollout_len": 30,
    },
    # 5
    {
        "name": "short_rollout",
        "lambda_l": 0.2,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 5,
        "goal_rollout_len": 10,
    },
    # 6
    {
        "name": "only_goal",
        "lambda_l": 0.0,
        "lambda_p": 0.0,
        "lambda_g": 1.0,
        "goal_distance": 5,
        "goal_rollout_len": 30,
    },
    # 7
    {
        "name": "goal+proximity",
        "lambda_l": 0.0,
        "lambda_p": 1.0,
        "lambda_g": 1.0,
        "goal_distance": 5,
        "goal_rollout_len": 30,
    },
]

for cfg in EXPERIMENTS[0:1]:
    sbatch_string = f"sbatch submit_train_mpur_goal.slurm \
    name={cfg['name']} \
    lambda_l={cfg['lambda_l']} \
    lambda_p={cfg['lambda_p']} \
    lamdba_g={cfg['lambda_g']} \
    goal_distance={cfg['goal_distance']} \
    goal_rollout_len={cfg['goal_rollout_len']}"

    os.system(f"set -k; {sbatch_string}")
