import argparse
import logging
import os
import sys
import time

from tqdm import tqdm

from code_search.twin.twin_train import load_examples
from metrices import metrics

sys.path.append("..")
sys.path.append("../../common")

import torch
from transformers import BertConfig
from common.models import TBertT, TBertS
from common.utils import evaluate_retrival, MODEL_FNAME, load_check_point, results_to_df, \
    format_batch_input_for_single_bert


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../data/code_search_net/python", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--model_path", default="./output/final_model", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--test_num", type=int,
                        help="The number of true links used for evaluation. The retrival task is build around the true links")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_dir", default="./evaluation/test", help="directory to store the results")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    parser.add_argument("--code_bert", default="microsoft/codebert-base", help="the base bert")
    args = parser.parse_args()
    return args


def test(args, model, eval_examples, batch_size=1000):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    retr_res_path = os.path.join(args.output_dir, "raw_result.csv")
    cache_file = "cached_single_test.dat"
    if args.overwrite or not os.path.isfile(cache_file):
        retrival_dataloader = eval_examples.get_batched_retrivial_task_dataloader(batch_size)
        torch.save(retrival_dataloader, cache_file)
    else:
        retrival_dataloader = torch.load(cache_file)

    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        nl_ids = batch[0]
        pl_ids = batch[1]
        labels = batch[2]
        with torch.no_grad():
            model.eval()
            inputs = format_batch_input_for_single_bert(batch, eval_examples, model)
            sim_score = model.get_sim_score(**inputs)
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), sim_score, labels.tolist()):
                res.append((n, p, prd, lb))

    df = results_to_df(res)
    df.to_csv(retr_res_path)
    m = metrics(df, output_dir=args.output_dir)

    pk = m.precision_at_K(3)
    best_f1, details = m.precision_recall_curve("pr_curve.png")
    map = m.MAP_at_K(3)
    mrr = m.MRR()

    return pk, best_f1, map, mrr


if __name__ == "__main__":
    args = get_eval_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    res_file = os.path.join(args.output_dir, "./raw_res.csv")

    cache_dir = os.path.join(args.data_dir, "cache")
    cached_file = os.path.join(cache_dir, "test_examples_cache.dat".format())

    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    model = TBertS(BertConfig(), args.code_bert)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))

    logger.info("model loaded")
    start_time = time.time()
    test_examples = load_examples(args.data_dir, data_type="test", model=model, overwrite=args.overwrite,
                                  num_limit=args.test_num)
    pk, best_f1, map, mrr = test(args, model, test_examples)
    exe_time = time.time() - start_time
    summary_path = os.path.join(args.output_dir, "summary.txt")
    summary = "\nprecision@3={}, best_f1 = {}, MAP={}, MRR={}\n".format(pk, best_f1, map, mrr, exe_time)
    with open(summary_path, 'w') as fout:
        fout.write(summary)
    logger.info("finished test")
