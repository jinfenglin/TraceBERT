import logging
import os
import sys
from typing import Dict

sys.path.append("..")
sys.path.append("../..")

from code_search.twin.twin_train import get_train_args
from trace_rnn.rnn_model import RNNTracer, create_emb_layer, LSTMEncoder
from trace_single.train_trace_single import load_examples
from common.data_structures import Examples, F_TOKEN

logger = logging.getLogger(__name__)
RNN_TK_ID = "RNN_TK_ID"


def update_rnn_feature(examples: Examples, model: RNNTracer):
    def __update_rnn_feature(index: Dict, encoder: LSTMEncoder):
        for id in index.keys():
            tokens = index[id][F_TOKEN]
            tk_id = encoder.token_to_ids(tokens)
            index[id][RNN_TK_ID] = tk_id

    __update_rnn_feature(examples.NL_index, model.nl_encoder)
    __update_rnn_feature(examples.PL_index, model.pl_encoder)

def update_rnn_embd(examples: Examples, model: RNNTracer):


def train(args):
    pass


def main():
    args = get_train_args()
    embd_info = create_emb_layer("./we/glove.6B.300d.txt")
    model = RNNTracer(hidden_dim=60, embd_info=embd_info)
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(train_dir, model=model, num_limit=args.train_num)
    valid_examples = load_examples(valid_dir, model=model, num_limit=args.valid_num)
    update_rnn_feature(train_examples, model)  # convert word into token id
    update_rnn_feature(valid_examples, model)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
