import gzip
import json
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertConfig

from model2.TBert import TBert


class CodeSearchNetReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.is_training = True

    def get_examples(self, type, num_limit=None):
        """
        :param type: train, valid, test
        :param num_limit:
        :return:
        """
        examples = []
        json_dir = os.path.join(self.data_dir, "final/jsonl")
        src_files = Path(os.path.join(json_dir, type)).glob('*.gz')
        for zfile in src_files:
            print("processing {}".format(zfile))
            with gzip.open(zfile, 'r') as fin:
                for line in fin.readlines():
                    if num_limit is not None:
                        if num_limit <= 0:
                            return examples
                        num_limit -= 1
                    jobj = json.loads(line)
                    code = jobj['code']
                    doc_str = jobj['docstring']
                    example = {
                        "NL": doc_str,
                        "PL": code
                    }
                    examples.append(example)
        return examples


class TBertProcessor:
    """
    We refer the features as the processed data that can be fed directly to TBert
    TODO move the functions out of this class as they can be reused to create dataset other than CodeSearchNet
    """

    def process_example(self, example, NL_tokenizer, PL_tokenizer, max_length):
        """
        return encoding for NL and PL. in tuple format ( NL_dict(), PL_dict())
        :param example:
        :return:
        """
        # return input_ids and attention mask.
        # attention_mask is important to filter out the paddings
        # https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
        nl_data = NL_tokenizer.encode_plus(example['NL'], max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True,
                                           return_token_type_ids=False)
        pl_data = PL_tokenizer.encode_plus(example['PL'], max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True,
                                           return_token_type_ids=False)
        nl = {
            "input_ids": nl_data['input_ids'],
            "attention_mask": nl_data['attention_mask']
        }
        pl = {
            "input_ids": pl_data['input_ids'],
            "attention_mask": pl_data['attention_mask']
        }
        return (nl, pl)

    def convert_examples_to_dataset(self, examples, NL_tokenizer, PL_tokenizer, is_training, threads=1):
        """

        :param examples:
        :param NL_tokenizer:
        :param PL_tokenizer:
        :param is_training: if it is training/evaluation then do not add label as it not exist.
        :param threads:
        :return:
        """
        pos_features = []
        neg_features = []
        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                self.process_example,
                NL_tokenizer=NL_tokenizer,
                PL_tokenizer=PL_tokenizer,
                max_length=512
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    desc="convert examples to positive features"
                )
            )
        if is_training:  # create negative instances for training purpose
            rel_index = defaultdict(set)
            NL_index = dict()  # find instance by id
            PL_index = dict()
            nl_cnt = 0
            pl_cnt = 0
            for f in tqdm(features, desc="assign ids to examples"):
                # assign id to the features
                nl_id = "N{}".format(nl_cnt)
                pl_id = "P{}".format(pl_cnt)
                f[0]['id'] = nl_id
                f[1]['id'] = pl_id
                NL_index[nl_id] = f[0]
                PL_index[pl_id] = f[1]
                rel_index[nl_id].add(pl_id)
                pos_features.append((f[0], f[1], 1))
                nl_cnt += 1
                pl_cnt += 1

            # create negative instances
            sample_time = 1
            for f in tqdm(features, desc="creating negative features"):
                nl, pl = f[0], f[1]
                nl_id, pl_id = nl['id'], pl['id']
                pos_pl_ids = rel_index[nl_id]
                for i in range(sample_time):
                    for c in PL_index.keys():
                        if c not in pos_pl_ids:
                            neg_features.append((NL_index[nl_id], PL_index[c], 0))
                # sample_pool = set(PL_index.keys()) - set(pos_pl_ids)
                #
                #
                # for _ in range(sample_time):
                #     neg_pl_id = random.choice(list(sample_pool))
                #     sample_pool.remove(neg_pl_id)  # do not oversample in current experiment setup
                #     neg_features.append((NL_index[nl_id], PL_index[neg_pl_id], 0))
        else:
            pos_features = features
        dataset = self.features_to_data_set(pos_features + neg_features, is_training)
        return dataset

    def features_to_data_set(self, features, is_training):
        # Convert to Tensors and build dataset， T-Bert will only need input_ids and will handle
        # the attention_mask etc automatically
        all_NL_input_ids = torch.tensor([f[0]['input_ids'] for f in features], dtype=torch.long)
        all_NL_attention_mask = torch.tensor([f[0]['attention_mask'] for f in features], dtype=torch.long)

        all_PL_input_ids = torch.tensor([f[1]['input_ids'] for f in features], dtype=torch.long)
        all_PL_attention_mask = torch.tensor([f[1]['attention_mask'] for f in features], dtype=torch.long)

        if is_training:
            all_labels = torch.tensor([f[2] for f in features], dtype=torch.long)
            dataset = TensorDataset(all_NL_input_ids, all_NL_attention_mask, all_PL_input_ids, all_PL_attention_mask,
                                    all_labels)
        else:
            dataset = TensorDataset(all_NL_input_ids, all_NL_attention_mask, all_PL_input_ids, all_PL_attention_mask)
        return dataset


if __name__ == "__main__":
    tb_process = CodeSearchNetReader(data_dir="G:\\Document\\code_search_net\\")
    examples = tb_process.get_examples(num_limit=None)
    model = TBert(BertConfig())
    data_set = TBertProcessor().convert_examples_to_dataset(examples, model.ntokenizer, model.ctokneizer,
                                                            is_training=True, threads=8)
    torch.save(data_set, "./dataset.dat")
    print(data_set)
