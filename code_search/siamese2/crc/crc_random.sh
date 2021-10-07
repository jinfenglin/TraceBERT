#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TBert_siamese2_random       # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/siamese2"
cd $root
source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
python siamese2_train.py \
    --data_dir ../data/code_search_net/python \
    --output_dir ./output \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --logging_steps 10 \
    --save_steps 10000 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 8 \
    --learning_rate 4e-5 \
    --valid_num 200 \
    --valid_step 10000 \
    --neg_sampling random
