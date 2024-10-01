#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-3%3
#SBATCH --time=10:00:00
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=80
#SBATCH --output=array_${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=array_${SLURM_ARRAY_TASK_ID}.err


module load R
Rscript splitting_script_cc.R ${SLURM_ARRAY_TASK_ID} 