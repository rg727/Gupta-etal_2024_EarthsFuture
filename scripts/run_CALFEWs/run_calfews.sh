#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-50%5
#SBATCH --time=10:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=80
#SBATCH --output=array_${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=array_${SLURM_ARRAY_TASK_ID}.err
#SBATCH --exclusive 



source /home/fs02/pmr82_0001/rg727/CALFEWS-main/.venv_conda_calfews/bin/activate 
python run_calfews.py "${SLURM_ARRAY_TASK_ID}"  