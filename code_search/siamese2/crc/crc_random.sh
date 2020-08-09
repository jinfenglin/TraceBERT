#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N TBert_siamese2_random       # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/siamese2"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/siamese2/task.log &
pip3 install -r /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/requirement.txt
chmod +x ./train_siamese2_crc_radnom.sh
./train_siamese2_crc_random.sh