#!/bin/bash
#SBATCH -J fg_pred_train
#SBATCH -p gpu
#SBATCH -w gpu1
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --error=%J.err
#SBATCH --output=%J.out

export I_MPI_ADJUST_REDUCE=3
cd $SLURM_SUBMIT_DIR

python end2end_10fold.py
