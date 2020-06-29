import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from common.data_structures import Examples
from common.metrices import metrics
from common.models import TwinBert

MODEL_FNAME = "t_bert.pt"
OPTIMIZER_FNAME = "optimizer.pt"
SCHED_FNAME = "scheduler.pt"
ARG_FNAME = "training_args.bin"

logger = logging.getLogger(__name__)


def format_batch_input(batch, examples, model):
    nl_ids, pl_ids, labels = batch[0], batch[1], batch[2]
    features = examples.id_pair_to_feature_pair(nl_ids, pl_ids)
    features = [t.to(model.device) for t in features]
    nl_in, nl_att, pl_in, pl_att = features
    inputs = {
        "text_ids": nl_in,
        "text_attention_mask": nl_att,
        "code_ids": pl_in,
        "code_attention_mask": pl_att,
    }
    return inputs


def write_tensor_board(tb_writer, data, step):
    for att_name in data.keys():
        att_value = data[att_name]
        tb_writer.add_scalar(att_name, att_value, step)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def save_examples(exampls, output_file):
    nl = []
    pl = []
    df = pd.DataFrame()
    for exmp in exampls:
        nl.append(exmp['NL'])
        pl.append(exmp['PL'])
    df['NL'] = nl
    df['PL'] = pl
    df.to_csv(output_file)


def save_check_point(model, ckpt_dir, args, optimizer, scheduler):
    logger.info("Saving checkpoint to %s", ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, MODEL_FNAME))
    torch.save(args, os.path.join(ckpt_dir, ARG_FNAME))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, OPTIMIZER_FNAME))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, SCHED_FNAME))


def load_check_point(model, ckpt_dir, optimizer, scheduler):
    logger.info("Loading checkpoint from {}".format(ckpt_dir))
    optmz_path = os.path.join(ckpt_dir, OPTIMIZER_FNAME)
    sched_path = os.path.join(ckpt_dir, SCHED_FNAME)
    model_path = os.path.join(ckpt_dir, MODEL_FNAME)

    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optmz_path))
    scheduler.load_state_dict(torch.load(sched_path))
    args = torch.load(os.path.join(ckpt_dir, ARG_FNAME))
    return {'model': model, "optimizer": optimizer, "scheduler": scheduler, "args": args}


def results_to_df(res: List[Tuple]) -> DataFrame:
    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    return df


def evaluate_classification(eval_examples: Examples, model: TwinBert, batch_size):
    eval_dataloader = eval_examples.random_neg_sampling_dataloader(batch_size=batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    num_correct = 0
    eval_num = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        inputs = format_batch_input(batch, eval_examples, model)
        with torch.no_grad():
            label = batch[2].to(model.device)
            outputs = model(**inputs)
            logit = outputs['logits']
            y_pred = logit.data.max(1)[1]
            batch_correct = y_pred.eq(label).long().sum().item()
            num_correct += batch_correct
            eval_num += y_pred.size()[0]

    accuracy = num_correct / eval_num
    tqdm.write("evaluate accuracy={}".format(accuracy))
    return accuracy


def evaluate_retrival(model, eval_examples: Examples, batch_size, res_dir):
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    retr_res_path = os.path.join(res_dir, "raw_result.csv")
    summary_path = os.path.join(res_dir, "summary.txt")
    retrival_dataloader = eval_examples.get_retrivial_task_dataloader(batch_size)
    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        nl_ids = batch[0]
        pl_ids = batch[1]
        labels = batch[2]
        nl_embd, pl_embd = eval_examples.id_pair_to_embd_pair(nl_ids, pl_ids)
        model.eval()
        with torch.no_grad():
            nl_embd.to(model.device)
            pl_embd.to(model.device)

            logits = model.cls(code_hidden=pl_embd, text_hidden=nl_embd)
            pred = torch.softmax(logits, 1).data.tolist()
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), pred, labels.tolist()):
                res.append((n, p, prd[1], lb))

    df = results_to_df(res)
    df.to_csv(retr_res_path)
    m = metrics(df, output_dir=res_dir)

    pk = m.precision_at_K(3)
    best_f1 = m.precision_recall_curve("pr_curve.png")
    map = m.MAP_at_K(3)

    summary = "precision@3={}, best_f1 = {}, MAP={}".format(pk, best_f1, map)
    tqdm.write(summary)
    with open(summary_path, 'w') as fout:
        fout.write(summary)
    return pk, best_f1, map
