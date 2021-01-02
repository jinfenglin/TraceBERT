# -*- coding: utf-8 -*-
import math
import os
import sys
import pandas as pd
from gensim.models import Word2Vec
import re

sys.path.append("..")
sys.path.append("../..")
from common.data_processing import CodeSearchNetReader

split_pattern = "\s|(?<!\d)[,.](?!\d)|//|\\n|\\\\|/|[\'=_\|]"


def split_art(art_list):
    res = []
    for a in art_list:
        if a != a:  # check nan
            continue
        words = [word for word in re.split(
            split_pattern, a) if word and word != ' ']
        res.append(words)
    return res


def read_data():
    res = []
    data_root = "../data/code_search_net/python"
    csn_reader = CodeSearchNetReader(data_root)
    for type in ["train", "test", "valid"]:
        raw_examples = csn_reader.get_examples(type=type, summary_only=True)
        for e in raw_examples:
            res.extend(split_art([e['NL']]))
            res.extend(split_art([e['PL']]))
    return res


if __name__ == "__main__":
    lines = read_data()
    print("load data with {} lines".format(len(lines)))
    model = Word2Vec(sentences=lines, size=100, window=5,
                     min_count=1, workers=4, iter=20)
    with open("./we/proj_embedding.txt", 'w', encoding='utf8') as fout:
        for w in model.wv.vocab:
            embd = " ".join([str(x) for x in model.wv[w]])
            fout.write("{} {}\n".format(w, embd))
