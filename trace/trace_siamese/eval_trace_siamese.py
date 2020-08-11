import logging
import os
import sys
import time

import torch
from transformers import BertConfig

sys.path.append("..")
sys.path.append("../../")

from code_search.twin.twin_eval import test
from code_search.twin.twin_eval import get_eval_args

from trace_single.train_trace_single import load_examples
from common.utils import MODEL_FNAME
from common.models import TBertI2

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

    model = TBertI2(BertConfig(), args.code_bert)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))
    # else:
    #     raise Exception("evaluation model not found")
    logger.info("model loaded")

    start_time = time.time()
    test_dir = os.path.join(args.data_dir, "test")
    test_examples = load_examples(test_dir, model=model, num_limit=args.test_num)
    test_examples.update_embd(model)
    m = test(args, model, test_examples, cache_file="cached_siamese_test.dat")
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
