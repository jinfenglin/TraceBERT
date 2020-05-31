# https://huggingface.co/blog/how-to-train
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset
import gzip
import json
import argparse
from transformers import BertTokenizer
import wget


def create_data(source, output):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    src_files = Path(source).glob('*.gz')
    if os.path.isfile(output):
        return
    with open(output, 'w', encoding='utf8') as fout:
        for zfile in src_files:
            print("processing {}".format(zfile))
            with gzip.open(zfile, 'r') as fin:
                for line in fin.readlines():
                    jobj = json.loads(line)
                    code = jobj['code']
                    doc_str = jobj['docstring']
                    input_ids = tokenizer.encode(code, doc_str, max_length=256)
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
            # tokenizer.prepare_for_model()
            # print(tokenizer.create_token_type_ids_from_sequences(lines[0]))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.data[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Java Data make script')
    parser.add_argument('-d', '--data_dir', type=str, default="G:/Document/code_search_net/")
    parser.add_argument('-l', '--language', type=str, default="java")
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(args.data_dir, args.language)):
        zip_file_name = '{}.zip'.format(args.language)
        print("Downloading data {}".format(zip_file_name))
        url = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}'.format(zip_file_name)
        out_path = os.path.join(args.data_dir, zip_file_name)
        wget.download(url, out_path)
        with ZipFile(out_path, 'r') as zipObj:
            zipObj.extractall(args.data_dir)
        os.remove(out_path)
        print("Finishing downloading...")

