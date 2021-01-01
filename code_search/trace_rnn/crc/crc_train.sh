#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TNN_t      # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/trace_rnn"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
#fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/siamese2/task.log &
#pip3 install -r /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/requirement.txt

python train_trace_rnn.py \
--data_dir ../data/code_search_net/python \
--output_dir ./output \
--logging_steps 10 \
--save_steps 10000 \
--num_train_epochs 8 \
--hidden_dim 60 \
--max_seq_len 80 \
--learning_rate 0.0001 \
--learning_rate 4e-5 \
--valid_num 200 \
--valid_step 10000
