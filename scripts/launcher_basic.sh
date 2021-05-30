#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH  --cpus-per-task=4
#SBATCH --mem=60000M
#SBATCH --mail-type=ALL,TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=arorakus@mila.quebec
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.out
###########################

set -ex
source ${HOME}/envs/lmpl/bin/activate 

export run_id=$(date '+%Y_%m_%d_%H_%M')
OUT_DIR=${HOME}/scratch/lmpl/results/${2}/$run_id/
export DISTRIBUTED=${DISTRIBUTED:-"false"};
export NUM_GPUS=${NUM_GPUS:-1};
export DEBUG=${DEBUG:-"true"};
export BATCH_SIZE=${BATCH_SIZE:-60};

tensorboard --logdir=${OUT_DIR}/log --host=0.0.0.0 &

allennlp train ${1} -s ${OUT_DIR} --include-package lmpl
