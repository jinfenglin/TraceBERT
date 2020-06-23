#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TBert_model3        # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/model3"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/model3/task.log &
#pip3 install -r ../requirement.txt

python3 train_model.py \
--data_dir ../model2/data/code_search_net/python \
--output_dir ./output \
--model_path ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 20 \
--save_steps 5000 \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--valid_num 100 \
--valid_step 1000 \
--gradient_accumulation_steps 16 \
--overwrite \
--resample_rate 6
