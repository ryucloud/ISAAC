#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J Social_Group_Loc_Training

# email error reports
#SBATCH --mail-user=babak2@illinois.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH -o loc-train-%j.out
#SBATCH -e loc-train-%j.err
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job

# Request runtime, memory, cores
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --mem-per-cpu=8000
#SBATCH --export=ALL

export PYTHONUNBUFFERED=FALSE
python Relevance_Classifier.py