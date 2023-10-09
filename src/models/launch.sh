#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 4
#$ -cwd
#$ -o ../out/std_$JOB_ID.out
#$ -e ../out/err_$JOB_ID.out

PYTHON_SCRIPT=$1

python $PYTHON_SCRIPT $2 $3