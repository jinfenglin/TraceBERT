import logging
import os
import sys
import time

sys.path.append("..")
sys.path.append("../../")

from code_search.twin.twin_eval import get_eval_args, test
from code_search.twin.twin_train import load_examples

import torch
from transformers import BertConfig
from common.models import TBertI2
from common.utils import MODEL_FNAME

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

    logger.info("model loaded")
    start_time = time.time()
    test_examples = load_examples(args.data_dir, data_type="test", model=model, overwrite=args.overwrite,
                                  num_limit=args.test_num)
    test_examples.update_embd(model)
    m = test(args, model, test_examples, "cached_siamese2_test")
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
