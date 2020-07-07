import argparse
import logging
import os
import sys

sys.path.append("..")
sys.path.append("../common")

import torch
from transformers import BertConfig

from model2.TBert_classify_train import load_examples
from common.models import TBert
from common.utils import evaluate_retrival, MODEL_FNAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./data/code_search_net/python", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--model_path", default="./output/final_model", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100,
                        help="The number of true links used for evaluation. The retrival task is build around the true links")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_dir", default="./eval_output", help="directory to store the results")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    res_file = os.path.join(args.output_dir, "./raw_res.csv")

    cache_dir = os.path.join(args.data_dir, "cache")
    cached_file = os.path.join(cache_dir, "classify_mdoel_eval_cache.dat".format())

    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    model = TBert(BertConfig())
    # args.model_path = os.path.join(args.model_path, MODEL_FNAME)
    # model.load_state_dict(torch.load(args.model_path))
    logger.info("model loaded")

    valid_examples = load_examples(args.data_dir, data_type="valid", model=model, num_limit=args.valid_num,
                                   overwrite=args.overwrite)
    valid_examples.update_embd(model)
    evaluate_retrival(model, valid_examples, args.per_gpu_eval_batch_size, args.output_dir)
