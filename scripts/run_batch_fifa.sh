#!/bin/bash

TIME_LIMIT=3600
MEM_LIMIT=16
INDICES=""

usage() {
  echo "Usage: $0 [-t time_limit_in_seconds] [-m memory_limit_in_GB] [-i indices]"
  echo "  -t : Time limit per phase in seconds (default: 3600)"
  echo "  -m : Memory limit in GB (default: 4)"
  echo "  -i : Job array indices to run, e.g. 0-4 or 0,2,4 (default: all 6)"
  echo ""
  echo "Sequences:"
  echo "  0 — For the fans          (5 1 2 3 4)"
  echo "  1 — Best broadcasting     (5 4 3 2 1)"
  echo "  2 — For the teams         (2 4 3 1 5)"
  echo "  3 — Eco                   (1 2 5 3 4)"
  echo "  4 — Help south africa     (4 3 5 2 1)"
  exit 1
}

while getopts "t:m:i:" opt; do
  case $opt in
    t) TIME_LIMIT=$OPTARG ;;
    m) MEM_LIMIT=$OPTARG ;;
    i) INDICES=$OPTARG ;;
    *) usage ;;
  esac
done

# Wall time = 5 phases × time_limit + 1h buffer
TOTAL=$((5 * TIME_LIMIT + 3600))
HOURS=$((TOTAL / 3600))
MINUTES=$(((TOTAL % 3600) / 60))
SECONDS=$((TOTAL % 60))
SLURM_TIME=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

ARRAY_OPTION=${INDICES:+--array=$INDICES}
ARRAY_OPTION=${ARRAY_OPTION:---array=0-4}

echo "Submitting FIFA scheduling jobs"
echo "  Sequences  : ${INDICES:-0-4}"
echo "  Time/phase : ${TIME_LIMIT}s"
echo "  Wall time  : ${SLURM_TIME}"
echo "  Memory     : ${MEM_LIMIT}G"

mkdir -p logs/fifa

sbatch --time=$SLURM_TIME \
       --mem=${MEM_LIMIT}G \
       --job-name=fifa_schedule \
       --output=logs/fifa/%A_%a.out \
       --export=ALL,TIME_LIMIT=$TIME_LIMIT \
       $ARRAY_OPTION \
       "$(dirname "$0")/run_fifa.sh"
