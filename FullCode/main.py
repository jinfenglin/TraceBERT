# https://huggingface.co/blog/how-to-train
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import gzip
import json
import argparse
from transformers import BertTokenizer
from subprocess import call


def create_data(source, output):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    src_files = Path(source).glob('*')
    for zfile in src_files:
        with gzip.open(zfile, 'r') as fin, open(output, 'w', encoding='utf8') as fout:
            for line in fin.readlines():
                jobj = json.loads(line)
                code = jobj['code']
                doc_str = jobj['docstring']
                input_ids = tokenizer.encode(code, doc_str, max_length=512)
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                fout.write(" ".join(tokens) + "\n")


class CodeDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = []
        file = './data/valid.txt' if evaluate else './data/train.txt'
        with open(file, encoding='utf8', errors='ignore') as fin:
            lines = fin.readlines()
            self.data = [tokenizer.convert_tokens_to_ids(line.split()) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.data[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Java Data make script')
    parser.add_argument('-d', '--data_dir', type=str, default="G:/Document/code_search_net/")
    parser.add_argument('-l', '--language', type=str, default="G:/Document/code_search_net/")
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(args.data_dir, args.language)):
        call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(args.language), '-P',
              args.data_dir, '-O', '{}.zip'.format(args.language)])
        call(['unzip', '{}.zip'.format(args.language)])
        call(['rm', '{}.zip'.format(args.language)])
    train_path = os.path.join(args.data_dir, "{}/final/jsonl/train".format(args.language))
    valid_path = os.path.join(args.data_dir, "{}/final/jsonl/valid".format(args.language))
    create_data(train_path, './data/trian.txt')
    create_data(valid_path, "./data/valid.txt")
