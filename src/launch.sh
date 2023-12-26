#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 16
#$ -cwd
#$ -o out/std_$JOB_ID.out
#$ -e out/err_$JOB_ID.out
#$ -q gpu7.q



export OMP_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export VECLIB_MAXIMUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

PYTHON_SCRIPT=$1

python $PYTHON_SCRIPT $2 $3
