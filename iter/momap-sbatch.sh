#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J momap_monitor
#SBATCH -o momap_monitor_%j.out
#SBATCH -e momap_monitor_%j.err

bash ./momap2.sh




