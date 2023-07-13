#!/bin/bash
#BSUB -P CSC499
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -q batch
#BSUB -J mldl_test_job
#BSUB -o /gpfs/alpine/csc499/scratch/vgurev/summit/transformer/summit_logs/job%J.out
#BSUB -e /gpfs/alpine/csc499/scratch/vgurev/summit/transformer/summit_logs/job%J.out
#BSUB -alloc_flags gpudefault


source ~/.bashrc
module load python/3.8-anaconda3
module load cuda
module load job-step-viewer
module load ums
module load ums-gen119
module load gcc/7.5.0
conda activate deepspeed


nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch)) 
head=${nodes[0]}                                                                   
export MASTER_ADDR=$head                                                           
export MASTER_PORT=29500 
# default from torch launcher 
echo $MASTER_ADDR   
echo $MASTER_PORT
echo $LD_LIBRARY_PATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/vgurev/extensions

jsrun -n1 -bpacked:7 -g6 -a6 -c42 -r1 python run_model_training.py
