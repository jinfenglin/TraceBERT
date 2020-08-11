#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TT_TPO        # trace train twin + post + online

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_twin"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online
