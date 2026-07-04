#!/bin/bash
#SBATCH -J train_iter
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --export=ALL

export I_MPI_ADJUST_REDUCE=3
cd $SLURM_SUBMIT_DIR

echo "Running fine-tuning job."
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "FIREFLY_ITER: ${FIREFLY_ITER:-1}"
echo "FIREFLY_TRAIN_SCRIPT: ${FIREFLY_TRAIN_SCRIPT:-CLLaMa_train_iter.py}"

python "${FIREFLY_TRAIN_SCRIPT:-CLLaMa_train_iter.py}"