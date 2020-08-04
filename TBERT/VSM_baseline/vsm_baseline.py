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
from transformers import BertConfig

from common.data_processing import CodeSearchNetReader
from metrices import metrics
from model2.TBert_classify_train import load_examples
from model2.VSM_baseline.IRs import VSM
import pandas as pd

from models import TBert

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
        nl_id = "{}".format(nl_cnt)
        pl_id = "{}".format(pl_cnt)
        f[0]['id'] = nl_id
        f[1]['id'] = pl_id
        NL_index[nl_id] = f[0]
        PL_index[pl_id] = f[1]
        rel_index[nl_id].add(pl_id)
        # pos_features.append((f[0], f[1], 1))
        nl_cnt += 1
        pl_cnt += 1

    for nl_cnt, nl_id in enumerate(NL_index):
        if nl_cnt > 100:
            break
        for pl_id in PL_index:
            if pl_id in rel_index[nl_id]:
                pos_features.append((NL_index[nl_id], PL_index[pl_id], 1))
            else:
                neg_features.append((NL_index[nl_id], PL_index[pl_id], 0))

    return pos_features, neg_features, NL_index, PL_index


def best_accuracy(data_frame, threshold_interval=1):
    df = DataFrame(columns=['acc', 'precision', 'recall', 'F1', 'tp', 'fp', 'tn', 'fn', 'threshold'])
    res = [x for x in zip(data_frame['pred'], data_frame['label'])]
    thresholds = [x for x in range(0, 100, threshold_interval)]
    with Pool(processes=8) as p:
        res = list(p.imap(partial(eval, res=res), thresholds))

    max_f1, out_p, out_re, out_thre = 0, 0, 0, 0
    for r in res:
        p = round(r[1], 3)
        re = round(r[2], 3)
        f = round(r[3], 3)
        thre = round(r[4] / 100, 2)
        tp, fp, tn, fn = int(r[5]), int(r[6]), int(r[7]), int(r[8])

        df = df.append({
            'acc': r[0],
            'precision': p,
            'recall': re,
            'F1': f,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'threshold': thre
        }, ignore_index=True)
        if f > max_f1:
            max_f1 = f
            out_p = p
            out_re = re
            out_thre = thre
    df.to_csv("f_score_eval.csv")
    # print("F1 = {}, precision={}, recall={}, thresold = {}".format(max_f1, out_p, out_re, out_thre))
    return max_f1, out_p, out_re, out_thre


def topN_RPF(data_frame, N):
    """
    count a hit if the true link is in the top N. Return Recall, Precision, F1 @N
    :param data_frame:
    :param N:
    :return:
    """
    res_dict = defaultdict(list)
    rel_cnt_at_N = 0
    N_total = 0
    total_rel = 0
    for _, r in tqdm(data_frame.iterrows(), desc="collect result into dictionary"):
        res_tuple = (r['s_id'], r['t_id'], r['pred'], r['label'])
        if res_tuple[3] == 1:
            total_rel += 1
        # ignore the 0 score to accelerating ranking
        if res_tuple[2] > 0:
            res_dict[r['s_id']].append(res_tuple)

    for s_id in tqdm(res_dict, "evalute each NL"):
        tuples = sorted(res_dict[s_id], key=lambda x: x[2], reverse=True)
        for i, t in enumerate(tuples):
            if i >= N:
                break
            N_total += 1
            if t[3] == 1:
                rel_cnt_at_N += 1
    success_rate = rel_cnt_at_N / len(res_dict) if len(res_dict) > 0 else 0
    # print("N = {}, SuccessRate={},Recall={}".format(N, round(success_rate, 3), round(recall, 3)))
    return success_rate


def MAP(data_frame):
    pass


def eval(threshold, res):
    t = threshold / 100.0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for pre, label in res:
        if pre > t:
            pre_label = 1
        else:
            pre_label = 0

        if pre_label == 1:
            if pre_label == label:
                tp += 1
            else:
                fp += 1
        else:
            if pre_label == label:
                tn += 1
            else:
                fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return (accuracy, precision, recall, f1, threshold, tp, fp, tn, fn)


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
    override = True
    valid_num = 200
    if not os.path.isfile(vsm_res_file) or override:
        data_dir = "../data/code_search_net/python"
        model = TBert(BertConfig())
        valid_examples = load_examples(data_dir, data_type="valid", model=model, num_limit=valid_num,
                                       overwrite=override)
        instances = valid_examples.get_retrivial_task_dataloader(batch_size=8).dataset
        doc_tokens = [x['tokens'].split() for x in valid_examples.PL_index.values()]
        vsm = VSM()
        vsm.build_model(doc_tokens)

        res = []
        for i, ins in tqdm(enumerate(instances)):
            s_id = ins[0]
            t_id = ins[1]
            label = ins[2]
            pred = vsm.get_link_scores(valid_examples.NL_index[ins[0]], valid_examples.PL_index[ins[1]])
            res.append((s_id, t_id, pred, label))
        df = pd.DataFrame()
        df['s_id'] = [x[0] for x in res]
        df['t_id'] = [x[1] for x in res]
        df['pred'] = [x[2] for x in res]
        df['label'] = [x[3] for x in res]
        df.to_csv(vsm_res_file)
    else:
        df = pd.read_csv(vsm_res_file)

    m = metrics(df, output_dir="/")
    pk = m.precision_at_K(3)
    best_f1, details = m.precision_recall_curve("pr_curve.png")
    map = m.MAP_at_K(3)

    summary = "\nprecision@3={}, best_f1 = {}, MAP={}\n".format(pk, best_f1, map)
    print(summary)
    # with open(summary_path, 'w') as fout:
    #     fout.write(summary)
    #     fout.write(str(details))
    # return pk, best_f1, map
