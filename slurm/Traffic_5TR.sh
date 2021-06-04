#!/bin/bash
#SBATCH --job-name=TF_Traffic
#SBATCH --output=/homes/svincenzi/TIME_SERIES/slurm/log/out.TF_Traffic.txt
#SBATCH --error=/homes/svincenzi/TIME_SERIES/slurm/log/err.TF_Traffic.txt
#SBATCH --open-mode=append
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source activate py_env2
module load cuda/10.0

export PYTHONPATH=/homes/svincenzi/TIME_SERIES


cd /homes/svincenzi/TIME_SERIES
srun python -u main.py --exp_name 'TF_Traffic!' --conf_file_path 'conf/traffic.yaml'
