#!/bin/bash
#SBATCH -J fg_gen_train
#SBATCH -p gpu
#SBATCH -w gpu1
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --error=%J.err
#SBATCH --output=%J.out

export I_MPI_ADJUST_REDUCE=3
cd $SLURM_SUBMIT_DIR

python CLLaMa_enhanced10.py
