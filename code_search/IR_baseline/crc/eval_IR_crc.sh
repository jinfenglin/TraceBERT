#!/bin/csh
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N IR_eval        # Specify job name

module load python/3.7.3
module load pytorch/1.1.0

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/IR_baseline"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
fsync /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/code_search/IR_baseline/task.log &
#pip3 install -r /afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/requirement.txt
chmod +x ./eval_IR.sh
./eval_IR.sh
