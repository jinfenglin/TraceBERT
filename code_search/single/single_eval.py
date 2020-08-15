import argparse
import logging
import os
import sys
import time

from torch.utils.data import DataLoader

sys.path.append("..")
sys.path.append("../../")
from tqdm import tqdm

from code_search.twin.twin_eval import get_eval_args
from code_search.twin.twin_train import load_examples

import torch
from transformers import BertConfig
from common.models import TBertS
from common.metrices import metrics
from common.utils import MODEL_FNAME, results_to_df, format_batch_input_for_single_bert


def test(args, model, eval_examples, chunk_size=1000):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    retr_res_path = os.path.join(args.output_dir, "raw_result.csv")
    cache_file = "cached_single_test.dat"
    if args.overwrite or not os.path.isfile(cache_file):
        chunked_retrivial_examples = eval_examples.get_chunked_retrivial_task_examples(
            chunk_query_num=args.chunk_query_num,
            chunk_size=chunk_size)
        torch.save(chunked_retrivial_examples, cache_file)
    else:
        chunked_retrivial_examples = torch.load(cache_file)
    retrival_dataloader = DataLoader(chunked_retrivial_examples, batch_size=args.per_gpu_eval_batch_size)

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
    return m


if __name__ == "__main__":
    args = get_eval_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    res_file = os.path.join(args.output_dir, "./raw_res.csv")

    cache_dir = os.path.join(args.data_dir, "cache")
    cached_file = os.path.join(cache_dir, "test_examples_cache.dat".format())

    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model = TBertS(BertConfig(), args.code_bert)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))

    logger.info("model loaded")
    start_time = time.time()
    test_examples = load_examples(args.data_dir, data_type="test", model=model, overwrite=args.overwrite,
                                  num_limit=args.test_num)
    m = test(args, model, test_examples)
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
    logger.info("finished test")
