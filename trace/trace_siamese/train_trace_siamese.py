import logging
import os
import sys



sys.path.append("..")
sys.path.append("../..")

from code_search.twin.twin_train import get_train_args, init_train_env
from code_search.twin.twin_train import train_with_neg_sampling, train, \
    logger
from trace_single.train_trace_single import load_examples

logger = logging.getLogger(__name__)


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='siamese2')
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(train_dir, model=model, num_limit=args.train_num)
    valid_examples = load_examples(valid_dir, model=model, num_limit=args.valid_num)
    train(args, train_examples, valid_examples, model, train_with_neg_sampling)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
