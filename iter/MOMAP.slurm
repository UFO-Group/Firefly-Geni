#!/bin/bash
#SBATCH -p cpu
#SBATCH -J MOMAP
#SBATCH -o _out.dat
#SBATCH -e _err.dat
#SBATCH -n 32

source /opt/soft/momap_2024A/MOMAP-2024A/env.sh

cd $SLURM_SUBMIT_DIR

momap -i momap.inp -n 32




