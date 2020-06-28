import gzip
import json
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count
from pathlib import Path
from typing import List, Set

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertConfig

from common.utils import exclude_and_sample
from model2.TBert import TBert


class CodeSearchNetReader:
    def __init__(self, data_dir, lang="python"):
        self.data_dir = data_dir
        self.is_training = True
        self.lang = lang

    def get_summary_from_docstring(self, docstring):
        summary = []
        for line in docstring.split("\n"):
            if self.lang == 'python':
                clean_line = line.strip("\n\t\r \"")
                if len(clean_line) == 0:
                    break
                if clean_line.startswith(":") or clean_line.startswith("TODO") \
                        or clean_line.startswith("Parameter") or clean_line.startswith("http"):
                    break
                summary.append(clean_line)
            else:
                summary.append(line)
        return " ".join(summary)

    def get_examples(self, type, num_limit=None, repos=[], summary_only=True):
        """
        :param type: train, valid, test
        :param num_limit: max number of examples
        :return:
        """
        examples = []
        doc_dup_check = defaultdict(list)
        json_dir = os.path.join(self.data_dir, "final/jsonl")
        src_files = Path(os.path.join(json_dir, type)).glob('*.gz')
        for zfile in src_files:
            print("processing {}".format(zfile))
            if num_limit is not None:
                if num_limit <= 0:
                    break
            with gzip.open(zfile, 'r') as fin:
                for line in fin.readlines():
                    if num_limit is not None:
                        if num_limit <= 0:
                            break
                    jobj = json.loads(line)
                    repo = jobj['repo']
                    if len(repos) > 0 and repo not in repos:
                        continue
                    code = jobj['code']
                    doc_str = jobj['docstring']
                    code = code.replace(doc_str, "")
                    if summary_only:
                        doc_str = self.get_summary_from_docstring(doc_str)
                    if len(doc_str.split()) < 10:  # abandon cases where doc_str is shorter than 10 tokens
                        continue
                    if num_limit:
                        num_limit -= 1
                    example = {
                        "NL": doc_str,
                        "PL": code
                    }
                    doc_dup_check[doc_str].append(example)
                    if num_limit and len(doc_dup_check[doc_str]) > 1:
                        num_limit += 1 + (len(doc_dup_check[doc_str]) == 2)

        for doc in doc_dup_check:
            if len(doc_dup_check[doc]) > 1:
                continue
            examples.extend(doc_dup_check[doc])
        return examples  # {nl:[pl]}


class DataConvert:
    """
    Data can be processed into 3 stages  examples -> DatasetCreater(features) -> dataset.

    This class manage the conversion
    1. examples -> features
    2. output of DatasetCreater -> dataset

    """

    def __init__(self):
        pass

    @staticmethod
    def clean_space(text):
        return " ".join(text.split())

    @staticmethod
    def process_example(example, NL_tokenizer, PL_tokenizer, max_length):
        """
        return encoding for NL and PL. in tuple format ( NL_dict(), PL_dict())
        :param example:
        :return:
        """
        # return input_ids and attention mask.
        # attention_mask is important to filter out the paddings
        # https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
        nl_data = NL_tokenizer.encode_plus(DataConvert.clean_space(example['NL']), max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True,
                                           return_token_type_ids=False)
        pl_data = PL_tokenizer.encode_plus(example['PL'], max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True,
                                           return_token_type_ids=False)
        nl = {
            "input_ids": nl_data['input_ids'],
            "attention_mask": nl_data['attention_mask'],
            'tokens': DataConvert.clean_space(example['NL'])
        }
        pl = {
            "input_ids": pl_data['input_ids'],
            "attention_mask": pl_data['attention_mask'],
            'tokens': example['PL'],
        }
        return (nl, pl)

    @staticmethod
    def index_exmaple_vecs(examples, NL_tokenizer, PL_tokenizer, threads=1):
        """
        read and create example dictionary
        :param examples:
        :param NL_tokenizer:
        :param PL_tokenizer:
        :param threads:
        :return:
        """
        threads = min(threads, os.cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                DataConvert.process_example,
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

        rel_index = defaultdict(set)
        NL_index = dict()  # find instance by id
        PL_index = dict()
        nl_id = 0
        pl_id = 0
        for f in tqdm(features, desc="assign ids to examples"):
            # assign id to the features

            f[0]['id'] = nl_id
            f[1]['id'] = pl_id
            NL_index[nl_id] = f[0]
            PL_index[pl_id] = f[1]
            rel_index[nl_id].add(pl_id)
            nl_id += 1
            pl_id += 1
        return NL_index, PL_index, rel_index

    @staticmethod
    def tuple_to_dataset(features):
        """
        Converting tuples in format (nl_info_dict, pl_info_dict, label) into classification dataset.
        For both nl and pl info dict, it should contain key 'id', and 'input_ids'
        :param features:
        :return:
        """
        # Convert to Tensors and build dataset， T-Bert will only need input_ids and will handle
        # the attention_mask etc automatically
        all_NL_ids = torch.tensor([int(f[0]['id']) for f in features], dtype=torch.long)
        all_NL_input_ids = torch.tensor([f[0]['input_ids'] for f in features], dtype=torch.long)
        all_NL_attention_mask = torch.tensor([f[0]['attention_mask'] for f in features], dtype=torch.long)

        all_PL_ids = torch.tensor([int(f[1]['id']) for f in features], dtype=torch.long)
        all_PL_input_ids = torch.tensor([f[1]['input_ids'] for f in features], dtype=torch.long)
        all_PL_attention_mask = torch.tensor([f[1]['attention_mask'] for f in features], dtype=torch.long)

        all_labels = torch.tensor([f[2] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_NL_ids, all_NL_input_ids, all_NL_attention_mask,
                                all_PL_ids, all_PL_input_ids, all_PL_attention_mask, all_labels)
        return dataset

    @staticmethod
    def hidden_state_to_dataset(features):
        all_NL_hidden = torch.stack([f[0]['embd'] for f in features])
        all_PL_hidden = torch.stack([f[1]['embd'] for f in features])
        all_labels = torch.tensor([f[2] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_NL_hidden, all_PL_hidden, all_labels)
        return dataset

    @staticmethod
    def triplet_to_dataset(features):
        pass


class DatasetCreater():
    """
    Organize the features in various format to accomplish different tasks.
    """

    def __init__(self):
        pass

    @staticmethod
    def random_balanced_tuples(examples):
        NL_index, PL_index, rel = examples["NL_index"], examples['PL_index'], examples['rel']
        pos_f, neg_f = [], []
        for nl_id in tqdm(rel, desc="creating dataset"):
            pos_pl_ids = rel[nl_id]
            for p_id in pos_pl_ids:
                pos_f.append((NL_index[nl_id], PL_index[p_id], 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(list(PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg_f.append((NL_index[nl_id], PL_index[n_id], 0))
        dataset = DataConvert.tuple_to_dataset(pos_f + neg_f)
        return dataset

    @staticmethod
    def offline_balanced_tuples(examples, models):
        pass

    @staticmethod
    def all_tuples(examples, model: TBert):
        """
        pairwised match between NL and PL. It will create a large population, therefore we use model to create embeddings
        and append them to NL_index and PL_index to avoid repeated computation.
        :param examples:
        :param model:
        :return:
        """
        pos_f, neg_f = [], []
        NL_index, PL_index, rel = examples["NL_index"], examples['PL_index'], examples['rel']
        if model:
            model.eval()
            with torch.no_grad():
                for nl_id in NL_index:
                    nl_feature = NL_index[nl_id]
                    input_ids = torch.tensor(nl_feature['input_ids']).view(-1, 1).to(model.device)
                    attention_mask = torch.tensor(nl_feature['attention_mask']).view(-1, 1).to(model.device)
                    NL_index[nl_id]['embd'] = model.create_nl_embed(input_ids, attention_mask)[0].to('cpu')

                for pl_id in PL_index:
                    pl_feature = PL_index[pl_id]
                    input_ids = torch.tensor(pl_feature['input_ids']).view(-1, 1).to(model.device)
                    attention_mask = torch.tensor(pl_feature['attention_mask']).view(-1, 1).to(model.device)
                    PL_index[pl_id]['embd'] = model.create_pl_embed(input_ids, attention_mask)[0].to('cpu')

        for nl_cnt, nl_id in enumerate(NL_index):
            for pl_id in PL_index:
                if pl_id in rel[nl_id]:
                    pos_f.append((NL_index[nl_id], PL_index[pl_id], 1))
                else:
                    neg_f.append((NL_index[nl_id], PL_index[pl_id], 0))
        dataset = DataConvert.hidden_state_to_dataset(pos_f + neg_f)
        return dataset

    @staticmethod
    def online_balanced_tuples(example_batch, models):
        pass
