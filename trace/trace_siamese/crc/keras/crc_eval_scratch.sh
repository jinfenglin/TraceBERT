#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N KE_IS      # eval trace siamese scratch

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_siamese"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"

python eval_trace_siamese.py \
--data_dir ../data/git_data/keras-team/keras \
--model_path ./output/keras/siamese2_scratch_0827/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name siamese2_scratch_0827
