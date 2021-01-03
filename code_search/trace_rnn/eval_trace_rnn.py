import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

from metrices import metrics

sys.path.append("..")
sys.path.append("../../")
from code_search.trace_rnn.train_trace_rnn import load_examples_for_rnn, update_rnn_embd, evaluate_rnn_retrival
from code_search.trace_rnn.rnn_model import RNNTracer, load_embd_from_file
from common.utils import MODEL_FNAME, ARG_FNAME, results_to_df
from code_search.twin.twin_eval import get_eval_args, test
import logging
import os
import torch

if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                                    and not args.no_cuda else "cpu")
    res_file = os.path.join(args.output_dir, "./raw_res.csv")
    cache_dir = os.path.join(args.data_dir, "cache")
    cached_file = os.path.join(cache_dir, "test_examples_cache.dat".format())
    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model_path = os.path.join(args.model_path, MODEL_FNAME)
    margs = torch.load(os.path.join(args.model_path, ARG_FNAME))

    embd_info = load_embd_from_file(margs.embd_file_path)
    model = RNNTracer(hidden_dim=margs.hidden_dim, embd_info=embd_info, embd_trainable=margs.is_embd_trainable,
                      max_seq_len=margs.max_seq_len,
                      is_no_padding=margs.is_no_padding if margs.is_no_padding is not None else False)
    model.load_state_dict(torch.load(model_path))
    logger.info("model loaded")

    test_examples = load_examples_for_rnn(
        args.data_dir, type='test', model=model, num_limit=args.test_num)
    update_rnn_embd(test_examples, model)
    m = test(args, model, test_examples, "cached_twin_test")
    m.write_summary(0)
    logger.info("finished test")
