#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N eval_post_SO_34k      # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_single"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/single/task.log &
#pip3 install -r /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/requirement.txt

python eval_trace_single.py \
--data_dir ../data/git_data/dbcli/pgcli \
#--model_path ./output/post_SO_34k/final_model \
--model_path ../pretrained_model/single_online_34000 \
--per_gpu_eval_batch_size 4 \
--exp_name single_post_SO_34k_at_0
