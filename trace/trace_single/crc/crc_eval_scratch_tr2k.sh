#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TE_SS_2k      # trace eval single scratch 4k model

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_single"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python eval_trace_single.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ./output/single_scratch/checkpoint-2000\
--per_gpu_eval_batch_size 4 \
--exp_name "eval_single_scratch_2k"
