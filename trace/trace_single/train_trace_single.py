import logging
import multiprocessing
import os
import sys

sys.path.append("..")
sys.path.append("../..")

from code_search.single.single_train import train_single_iteration
from code_search.twin.twin_train import get_train_args, init_train_env, train
from data_process import __read_artifacts
from common.data_structures import Examples
from common.models import TBertT, TBertI, TBertI2

logger = logging.getLogger(__name__)


def read_OSS_examples(data_dir):
    commit_file = os.path.join(data_dir, "commit_file")
    issue_file = os.path.join(data_dir, "issue_file")
    link_file = os.path.join(data_dir, "link_file")
    examples = []
    issues = __read_artifacts(issue_file, "issue")
    commits = __read_artifacts(commit_file, "commit")
    links = __read_artifacts(link_file, "link")
    issue_index = {x.issue_id: x for x in issues}
    commit_index = {x.commit_id: x for x in commits}
    for lk in links:
        iss = issue_index[lk[0]]
        cm = commit_index[lk[1]]
        # join the tokenized content
        iss_text = iss.desc + " " + iss.comments
        cm_text = cm.summary + " " + cm.diffs
        example = {
            "NL": iss_text,
            "PL": cm_text
        }
        examples.append(example)
    return examples


def load_examples(data_dir, model, num_limit):
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    logger.info("Creating examples from dataset file at {}".format(data_dir))
    raw_examples = read_OSS_examples(data_dir)
    if num_limit:
        raw_examples = raw_examples[:num_limit]
    examples = Examples(raw_examples)
    if isinstance(model, TBertT) or isinstance(model, TBertI2) or isinstance(model, TBertI):
        examples.update_features(model, multiprocessing.cpu_count())
    return examples


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='single')
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(train_dir, model=model, num_limit=args.train_num)
    valid_examples = load_examples(valid_dir, model=model, num_limit=args.valid_num)
    train(args, train_examples, valid_examples, model, train_single_iteration)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
