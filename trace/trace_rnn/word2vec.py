import os
import pandas as pd
from gensim.models import Word2Vec
import re


def read_data():
    split_pattern = "(\s|(?<!\d)[,.](?!\d)|//|\\n|\\\\|/|[\'=_\|])"
    res = []
    data_root = "../data/git_data"
    for proj in ['dbcli/pgcli', 'keras-team/keras', 'pallets/flask']:
        for part in ['train', 'valid', 'test']:
            cm = pd.read_csv(os.path.join(data_root, proj, part, "commit_file"))
            iss = pd.read_csv(os.path.join(data_root, proj, part, "issue_file"))
            res.extend([re.split(split_pattern, x) for x in cm['diff']])
            res.extend([re.split(split_pattern, x) for x in cm['summary']])
            res.extend([re.split(split_pattern, x) for x in iss['issue_comments']])
            res.extend([re.split(split_pattern, str(x)) for x in iss['issue_desc']])
    return res


if __name__ == "__main__":
    lines = read_data()
    print("load data with {} lines".format(len(lines)))
    model = Word2Vec(sentences=lines, size=100, window=5, min_count=1, workers=4, iter=20)
    model.wv.save("proj_vocab.wordvectors")
    with open("./we/proj_embedding.txt", 'w', encoding='utf8') as fout:
        for w in model.wv.vocab:
            embd = " ".join([str(x) for x in model.wv['to']])
            fout.write("{} {}\n".format(w, embd))
