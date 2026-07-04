#!/bin/bash
#BATCH -J orca-test
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --error=%J.err
#SBATCH --output=%J.out

export RSH_COMMAND="ssh"
export I_MPI_ADJUST_REDUCE=3
cd $SLURM_SUBMIT_DIR

export PATH=/opt/soft/OpenMpi411/bin:/opt/soft/orca-5.0.1:$PATH
export LD_LIBRARY_PATH=/opt/soft/OpenMpi411/lib:/opt/soft/orca-5.0.1:$LD_LIBRARY_PATH
export orcapath=/opt/soft/orca-5.0.1

for INPUT in *.inp
do
        echo "ORCA job start at" 'date'
        time $orcapath/orca $INPUT > log
done




