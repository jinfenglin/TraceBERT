#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N RNN_E_FL     #  trace eavl RNN Flask

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_rnn"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python eval_trace_rnn.py \
--data_dir ../data/git_data/pallets/flask \
--model_path ./output/flask/checkpoint-6000 \
--per_gpu_eval_batch_size 4 \
--exp_name flask
