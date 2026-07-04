#!/bin/bash
#SBATCH -J pytorch
#SBATCH -p gpu
#SBATCH -w gpu4
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --error=%J.err
#SBATCH --output=%J.out

export I_MPI_ADJUST_REDUCE=3
cd $SLURM_SUBMIT_DIR

python gen_est_049_mask.py --est 0.05 --sa 2.5