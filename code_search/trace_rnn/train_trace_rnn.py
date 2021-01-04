from typing import Dict
import re
import os
import logging
import datetime
import argparse
import torch
from torch import Tensor
from tqdm import tqdm
import sys

sys.path.append("..")
sys.path.append("../..")

from code_search.twin.twin_train import train
from common.metrices import metrics
from common.utils import write_tensor_board, save_check_point, evaluate_classification, evaluate_retrival, \
    results_to_df, format_rnn_batch_input
from code_search.trace_rnn.rnn_model import RNNTracer, create_emb_layer, RNNEncoder, load_embd_from_file
from common.data_structures import Examples, F_TOKEN, F_INPUT_ID, F_EMBD
from common.data_processing import CodeSearchNetReader

logger = logging.getLogger(__name__)
RNN_TK_ID = F_INPUT_ID
RNN_EMBD = F_EMBD

rnn_split_pattern = "\s|(?<!\d)[,.](?!\d)|//|\\n|\\\\|/|[\'=_\|]"


def load_examples_for_rnn(data_dir, type, model, num_limit):
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    logger.info("Creating examples from dataset file at {}".format(data_dir))
    csn_reader = CodeSearchNetReader(data_dir)
    raw_examples = csn_reader.get_examples(type, summary_only=True)
    if num_limit:
        raw_examples = raw_examples[:num_limit]

    for e in raw_examples:
        e['NL'] = " ".join(re.split(rnn_split_pattern, e['NL']))
        e['PL'] = " ".join(re.split(rnn_split_pattern, e['PL']))
    examples = Examples(raw_examples)
    update_rnn_feature(examples, model)
    return examples


def update_rnn_feature(examples: Examples, model: RNNTracer):
    def __update_rnn_feature(index: Dict, encoder: RNNEncoder):
        for id in index.keys():
            tokens = index[id][F_TOKEN].split()
            tk_id = encoder.token_to_ids(tokens)
            index[id][RNN_TK_ID] = tk_id

    __update_rnn_feature(examples.NL_index, model.nl_encoder)
    __update_rnn_feature(examples.PL_index, model.pl_encoder)


def update_rnn_embd(examples: Examples, model: RNNTracer):
    def __update_rnn_embd(index: Dict, sub_model: RNNEncoder):
        for id in tqdm(index.keys(), desc="update RNN embedding"):
            tk_id = index[id][RNN_TK_ID].view(1, -1)
            embd = sub_model(tk_id.to(model.device))[0]
            index[id][RNN_EMBD] = embd.to('cpu')

    with torch.no_grad():
        model.eval()
        __update_rnn_embd(examples.NL_index, model.nl_encoder)
        __update_rnn_embd(examples.PL_index, model.pl_encoder)


def get_rnn_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./data", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument(
        "--model_path", default=None, type=str,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=50,
                        help="obtain validation accuracy every given steps")
    parser.add_argument("--train_num", type=int, default=None,
                        help="number of instances used for training")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the cached data")
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument(
        "--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--save_steps", type=int, default=3000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="maximal input sequence length")
    parser.add_argument(
        "--embd_file_path", type=str, help="the path of word embdding file")
    parser.add_argument(
        "--is_embd_trainable", default=False, action='store_true', help="whether the embedding is trainable")
    parser.add_argument("--hidden_dim", type=int, default=60,
                        help="hidden state dimension")
    parser.add_argument("--exp_name", type=str,
                        help="ID for this execution, RNN training must specify a name as a quick fix")
    parser.add_argument("--is_no_padding", default=False, action='store_true',
                        help="if do not have padding then the batch size is always 1")
    parser.add_argument("--neg_sampling", default='random', choices=['random', 'online', 'offline'],
                        help="Negative sampling strategy we apply for constructing dataset. ")
    parser.add_argument("--rnn_type", default='bi_gru', choices=['bi_gru', 'lstm'],
                        help="Type of RNN layer in TraceNN ")
    parser.add_argument(
        "--hard_ratio", default=0.5, type=float,
        help="The ration of hard negative examples in a batch during negative sample mining"
    )
    return parser.parse_args()


def train_rnn_iter(args, model: RNNTracer, train_examples: Examples, valid_examples: Examples, optimizer,
                   scheduler, tb_writer, step_bar, skip_n_steps):
    tr_loss, tr_ac = 0, 0
    batch_size = args.per_gpu_train_batch_size
    cache_file = "cached_rnn_random_neg_sample_epoch_{}.dat".format(
        args.epochs_trained)

    if args.neg_sampling == "random":
        if args.overwrite or not os.path.isfile(cache_file):
            train_dataloader = train_examples.random_neg_sampling_dataloader(
                batch_size=batch_size)
            torch.save(train_dataloader, cache_file)
        else:
            train_dataloader = torch.load(cache_file)
    elif args.neg_sampling == "online":
        # we provide only positive cases and will create negative in the batch processing
        train_dataloader = train_examples.online_neg_sampling_dataloader(
            batch_size=int(batch_size / 2))
    else:
        raise Exception(
            "{} neg_sampling is not recoginized...".format(args.neg_sampling))

    for step, batch in enumerate(train_dataloader):
        if skip_n_steps > 0:
            skip_n_steps -= 1
            continue

        if args.neg_sampling == "online":
            batch = train_examples.make_online_neg_sampling_batch(
                batch, model, args.hard_ratio)

        model.train()
        labels = batch[2].to(model.device)
        inputs = format_rnn_batch_input(batch, train_examples, model)
        inputs['label'] = labels
        outputs = model(**inputs)
        loss = outputs['loss']
        logit = outputs['logits']
        y_pred = logit.data.max(1)[1]
        tr_ac += y_pred.eq(labels).long().sum().item()

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            try:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        else:
            loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            args.global_step += 1
            step_bar.update()

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                tb_data = {
                    'lr': scheduler.get_last_lr()[0],
                    'acc': tr_ac / args.logging_steps / (
                            args.train_batch_size * args.gradient_accumulation_steps),
                    'loss': tr_loss / args.logging_steps
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
                tr_loss = 0.0
                tr_ac = 0.0

            # Save model checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and args.global_step % args.save_steps == 0:
                # step invoke checkpoint writing
                ckpt_output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}".format(args.global_step))
                save_check_point(model, ckpt_output_dir,
                                 args, optimizer, scheduler)

            if args.valid_step > 0 and args.global_step % args.valid_step == 1:
                # step invoke validation
                update_rnn_embd(valid_examples, model)
                valid_accuracy, valid_loss = evaluate_rnn_classification(valid_examples, model,
                                                                         args.per_gpu_eval_batch_size,
                                                                         "evaluation/rnn/runtime_eval")
                pk, best_f1, map = evaluate_rnn_retrival(model, valid_examples, args.per_gpu_eval_batch_size,
                                                         "evaluation/rnn/runtime_eval")
                tb_data = {
                    "valid_accuracy": valid_accuracy,
                    "valid_loss": valid_loss,
                    "precision@3": pk,
                    "best_f1": best_f1,
                    "MAP": map
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
        args.steps_trained_in_current_epoch += 1
        if args.max_steps > 0 and args.global_step > args.max_steps:
            break


def evaluate_rnn_classification(eval_examples, model: RNNTracer, batch_size, output_dir, append_label=True):
    eval_dataloader = eval_examples.random_neg_sampling_dataloader(
        batch_size=batch_size)
    clsfy_res = []
    num_correct = 0
    eval_num = 0
    eval_loss = 0
    for batch in tqdm(eval_dataloader, desc="RNN Classify Evaluating"):
        with torch.no_grad():
            model.eval()
            labels = batch[2].to(model.device)
            inputs = format_rnn_batch_input(batch, eval_examples, model)
            inputs['label'] = labels
            outputs = model(**inputs)
            logit = outputs['logits']
            if append_label:
                loss = outputs['loss'].item()
                eval_loss += loss
            y_pred = logit.data.max(1)[1]
            # y_pred = torch.squeeze(logit.data, 0).max(1)[1]
            batch_correct = y_pred.eq(labels).long().sum().item()
            num_correct += batch_correct
            eval_num += y_pred.size()[0]
            clsfy_res.append((y_pred, labels, batch_correct))

    accuracy = num_correct / eval_num
    eval_loss = eval_loss / len(eval_dataloader)
    tqdm.write("\nevaluate accuracy={}\n".format(accuracy))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    res_file = os.path.join(output_dir, "raw_rnn_classify_res.txt")
    with open(res_file, 'w') as fout:
        for res in clsfy_res:
            fout.write(
                "pred:{}, label:{}, num_correct:{}\n".format(str(res[0].tolist()), str(res[1].tolist()), str(res[2])))
    return accuracy, eval_loss


def _id_to_embd(id_tensor: Tensor, index):
    embds = []
    for id in id_tensor.tolist():
        embds.append(index[id][RNN_EMBD])
    embd_tensor = torch.stack(embds)
    return embd_tensor


def evaluate_rnn_retrival(model: RNNTracer, eval_examples, batch_size, res_dir):
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    retr_res_path = os.path.join(res_dir, "raw_result.csv")
    summary_path = os.path.join(res_dir, "summary.txt")
    retrival_dataloader = eval_examples.get_retrivial_task_dataloader(
        batch_size)
    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        nl_ids = batch[0]
        pl_ids = batch[1]
        labels = batch[2]
        nl_embd, pl_embd = _id_to_embd(nl_ids, eval_examples.NL_index), _id_to_embd(
            pl_ids, eval_examples.PL_index)

        with torch.no_grad():
            model.eval()
            nl_embd = nl_embd.to(model.device)
            pl_embd = pl_embd.to(model.device)
            sim_score = model.get_sim_score(
                text_hidden=nl_embd, code_hidden=pl_embd)
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), sim_score, labels.tolist()):
                res.append((n, p, prd, lb))
    df = results_to_df(res)
    df.to_csv(retr_res_path)
    m = metrics(df, output_dir=res_dir)
    m.write_summary(0)

    pk = m.precision_at_K(3)
    best_f1, best_f2, details, threshold = m.precision_recall_curve(
        "pr_curve.png")
    map = m.MAP_at_K(3)
    return pk, best_f1, map


def main():
    args = get_rnn_train_args()
    if args.is_no_padding:
        # args.gradient_accumulation_steps = 1
        args.per_gpu_eval_batch_size = 1
        args.per_gpu_train_batch_size = 1

    args.exp_name = "{}_{}".format(
        args.exp_name, datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    # embd_info = create_emb_layer("./we/glove.6B.300d.txt")
    embd_info = load_embd_from_file(args.embd_file_path)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    model = RNNTracer(hidden_dim=args.hidden_dim, embd_info=embd_info, embd_trainable=args.is_embd_trainable,
                      max_seq_len=args.max_seq_len, is_no_padding=args.is_no_padding, rnn_type= args.rnn_type)
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    model.to(args.device)
    model.device = args.device
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    train_examples = load_examples_for_rnn(
        args.data_dir, type="train", model=model, num_limit=args.train_num)
    valid_examples = load_examples_for_rnn(
        args.data_dir, type="valid", model=model, num_limit=args.valid_num)
    logger.info("Training started")
    train(args, train_examples, valid_examples, model, train_rnn_iter)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
