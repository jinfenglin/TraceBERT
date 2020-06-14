import os
import random
import re
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count
from nltk.tokenize import word_tokenize

import many_stop_words
from pandas import DataFrame
from tqdm import tqdm

from model2 import CodeSearchNetReader
from model2.VSM_baseline.IRs import VSM
import pandas as pd

stop_words = many_stop_words.get_stop_words('en')


def preprocess(text):
    clean_text = re.sub("\W+", " ", text).lower()
    tokens = word_tokenize(clean_text)
    tokens = [tk for tk in tokens if tk not in stop_words]
    res = []
    for tk in tokens:
        res.extend(tk.split("_"))
    return res


def process_example(example):
    """
    return encoding for NL and PL. in tuple format ( NL_dict(), PL_dict())
    :param example:
    :return:
    """
    nl_text = example['NL']
    pl_text = example['PL']
    nl = {'tokens': preprocess(nl_text), 'raw': nl_text}
    pl = {'tokens': preprocess(pl_text), 'raw': pl_text}
    return (nl, pl)


def convert_examples_to_dataset(examples, threads=1):
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
            process_example,
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
    pl_id_list = list(PL_index.keys())
    for f in tqdm(features, desc="creating negative features"):
        nl, pl = f[0], f[1]
        nl_id, pl_id = nl['id'], pl['id']
        pos_pl_ids = rel_index[nl_id]
        retry = 3
        sample_time = 1
        while sample_time > 0:
            neg_pl_id = pl_id_list[random.randint(0, len(pl_id_list) - 1)]
            if neg_pl_id not in pos_pl_ids:
                neg_features.append((NL_index[nl_id], PL_index[neg_pl_id], 0))
                retry = 3
                sample_time -= 1
            else:
                retry -= 1
                if retry == 0:
                    break
    return pos_features, neg_features


def best_accuracy(data_frame, threshold_interval=1):
    res = [x for x in zip(data_frame['pred'], data_frame['label'])]
    thresholds = [x for x in range(0, 100, threshold_interval)]
    with Pool(processes=8) as p:
        acc = list(p.imap(partial(eval, res=res), thresholds))
    print(acc)
    return max(acc)


def eval(threshold, res):
    t = threshold / 100.0
    cnt = 0
    for pre, label in res:
        if pre > t:
            pre_label = 1
        else:
            pre_label = 0
        cnt += pre_label == label
    return cnt / len(res)


def debug_instnace(instances):
    cm_df = DataFrame()
    source_df = DataFrame()
    target_df = DataFrame()
    source = []
    target = []
    cm = []
    cm_cnt = []
    for i in tqdm(instances):
        ctk = list(set(i[0]['tokens']).intersection(set(i[1]['tokens'])))
        cm.append(ctk)
        cm_cnt.append(len(ctk))
        source.append(i[0]['raw'])
        target.append(i[1]['raw'])
    cm_df['common_count'] = cm_cnt
    cm_df['common_tokens'] = cm
    source_df['raw'] = source
    target_df['raw'] = target
    cm_df.to_csv("common_tokens.csv")
    source_df.to_csv('debug_source.csv')
    target_df.to_csv('debug_target.csv')


if __name__ == "__main__":
    vsm_res_file = "vsm_res.csv"
    if os.path.isfile(vsm_res_file):
        data_dir = "../data/code_search_net/python"
        csr = CodeSearchNetReader(data_dir)
        examples = csr.get_examples('valid')
        pos, neg = convert_examples_to_dataset(examples)
        instances = pos + neg
        debug_instnace(instances)
        doc_tokens = [x[1]['tokens'] for x in pos]
        # doc_tokens.extend(x[0]['tokens'] for x in instances)
        vsm = VSM()
        vsm.build_model(doc_tokens)

        res = []
        for i, ins in tqdm(enumerate(instances)):
            label = ins[2]
            pred = vsm.get_link_scores(ins[0], ins[1])
            res.append((pred, label))
        df = pd.DataFrame()
        df['pred'] = [x[0] for x in res]
        df['label'] = [x[1] for x in res]
        df.to_csv(vsm_res_file)
    else:
        df = pd.read_csv(vsm_res_file)
    print("evaluating...")
    ba = best_accuracy(df, threshold_interval=1)
    print(ba)
