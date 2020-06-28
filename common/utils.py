import logging
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import pandas as pd

MODEL_FNAME = "t_bert.pt"
OPTIMIZER_FNAME = "optimizer.pt"
SCHED_FNAME = "scheduler.pt"
ARG_FNAME = "training_args.bin"

logger = logging.getLogger(__name__)


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
    args = torch.load(ARG_FNAME)
    return {'model': model, "optimizer": optimizer, "scheduler": scheduler, "args": args}


def exclude_and_sample(sample_pool, exclude, num):
    """"""
    for id in exclude:
        sample_pool.remove(id)
    selected = random.sample(sample_pool, num)
    return selected

def clean_space(text):
    return " ".join(text.split())