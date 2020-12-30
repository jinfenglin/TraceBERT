#!/bin/csh
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N w2v     # train word embedding

module load python/3.7.3

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/trace/trace_rnn"
cd $root

source "/afs/crc.nd.edu/user/j/jlin6/projects/ICSE2020/venv/bin/activate.csh"
pip3 install gensim
python word2vec.py
