# https://huggingface.co/blog/how-to-train
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import gzip
import json

from transformers import BertTokenizer


def create_data(source, output):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.isfile(output):
        # src_files = Path(data_dir).glob("train\\java_train_*.jsonl.gz")
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
        self.data = self.data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.data[i])


if __name__ == '__main__':
    create_data("G:\\Document\\code_search_net\\java\\final\jsonl\\train", './data/trian.txt')
    create_data("G:\\Document\\code_search_net\\java\\final\\jsonl\\valid", "./data/valid.txt")
