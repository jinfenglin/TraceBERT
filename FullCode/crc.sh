#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N bert_for_code        # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/FullCode"
cd $root
source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/FullCode/task.log &
#pip3 install -r ../requirement.txt
python3 main.py --language java --data_dir '/afs/crc.nd.edu/user/j/jlin6/data/code_search_net' >&crc.log
python3 run_language_modeling.py --output_dir ./model/full_code --model_type bert --mlm --tokenizer_name bert-base-uncased \
--do_train --do_eval --learning_rate 1e-4 --num_train_epochs 100 --save_total_limit 10 --save_steps 2000 \
--per_gpu_train_batch_size 16 --overwrite_output_dir --seed 42 --eval_data_file ./data/train.txt >&crc.log
