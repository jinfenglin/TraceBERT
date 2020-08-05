import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


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
                    jobj = json.loads(str(line, encoding='utf-8'))
                    repo = jobj['repo']
                    if len(repos) > 0 and repo not in repos:
                        continue
                    # code = jobj['code']
                    code = ' '.join([format_str(token) for token in jobj['code_tokens']])
                    # doc_str = jobj['docstring']
                    doc_str = ' '.join(jobj['docstring_tokens'])
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
