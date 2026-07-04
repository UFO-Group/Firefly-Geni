#!/bin/bash
#SBATCH  -p  cpu
#SBATCH  -N  1
#SBATCH  -n  32

export PGI_FASTMATH_CPU=sandybridge
export g16root=/opt/soft
export GAUSS_SCRDIR=/opt/soft/g16/scratch
source /opt/soft/g16/bsd/g16.profile

g16 $1 >log



