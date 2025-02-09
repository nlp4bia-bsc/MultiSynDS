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
#SBATCH --time=2:00:00
#SBATCH --exclusive

source .venv/bin/activate

python scripts/llm_judge/llama_3B_inst_eval.py scripts/llm_judge/llama_3B_inst_eval.py
