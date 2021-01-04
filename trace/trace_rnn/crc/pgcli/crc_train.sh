#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N rnn_t_pg      # trace train single + post + online

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_rnn"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python train_trace_rnn.py \
--data_dir ../data/git_data/dbcli/pgcli \
--output_dir ./output \
--embd_file_path ./we/proj_embedding.txt \
--exp_name pgcli \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru

