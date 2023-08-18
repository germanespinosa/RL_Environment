#!/bin/bash
#SBATCH -A #####
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=40G
#SBATCH --job-name="Shuo_han\${SLURM_ARRAY_TASK_ID}" ## use the task id in the name of the job
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=#####  ## your email



echo "Running on a single CPU core"

source ~/.bashrc
conda activate torchenv
python sac_dense.py

date