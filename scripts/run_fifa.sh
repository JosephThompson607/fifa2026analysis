#!/bin/bash
#SBATCH --account=def-jmendoza

# Load environment
module load python/3.13
module load gurobi
source $HOME/envs/venv-thesis/bin/activate

# Sequences indexed 0-5: each is a space-separated priority order
SEQUENCES=(
    "5 1 2 3 4"   # 0 — For the fans
    "5 4 3 2 1"   # 1 — Best games for broadcasting
    "2 4 3 1 5"   # 2 — For the teams
    "1 2 5 3 4"   # 3 — Eco
    "4 3 5 2 1"   # 4 — Help south africa
)

NAMES=(
    "for_the_fans"
    "best_broadcasting"
    "for_the_teams"
    "eco"
    "help_south_africa"
)

SEQ=${SEQUENCES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running sequence $SLURM_ARRAY_TASK_ID ($NAME): $SEQ"

cd $HOME/Thesis/fifa2026analysis

python model.py -t $TIME_LIMIT -p $SEQ
