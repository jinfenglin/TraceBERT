#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N FE_SP     # Flask eval single post

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_single"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python eval_trace_single.py \
--data_dir ../data/git_data/pallets/flask \
--model_path ./output/flask/single_transfer_0826/checkpoint-4001 \
--per_gpu_eval_batch_size 4 \
--exp_name single_transfer_0826
