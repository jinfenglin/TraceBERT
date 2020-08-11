#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N eval_TT_S      # trace Twin from scratch

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_twin"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python eval_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ./output/twin_scratch/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name twin_scratch
