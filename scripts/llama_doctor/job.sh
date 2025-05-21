#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs_ce_train/ce_train_%j.out
#SBATCH --error=logs_ce_train/ce_train_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --exclusive

module load anaconda

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

conda activate msds

$CONDA_PREFIX/bin/python scripts/llama_doctor/main.py