import argparse
import logging
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import BertConfig, get_linear_schedule_with_warmup

sys.path.append("..")
from model2 import CodeSearchNetReader, TBertProcessor
from model2.VSM_baseline.vsm_baseline import best_accuracy, topN_RPF
from model2.exec_TBert_experiment import save_check_point
from model3.TBert_2 import TBert2

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./data", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument(
        "--model_path", default=None, type=str, required=True,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=50,
                        help="obtain validation accuracy every given steps")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--resample_rate", default=1, type=int, help="Oversample rate for positive examples, "
                                                                     "negative examples will match the number")
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        args.n_gpu = 1
    args.device = device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    model = TBert2(BertConfig())
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    valid_dataset = load_dataset(args.data_dir, "valid",
                                 model.ntokenizer, model.ctokneizer,
                                 num_limit=args.valid_num, overwrite=args.overwrite)
    train_dataset = load_dataset(args.data_dir, "train",
                                 model.ntokenizer, model.ctokneizer,
                                 num_limit=None, overwrite=args.overwrite, resample_rate=args.resample_rate)

    global_step, tr_loss = train(args, train_dataset, valid_dataset, model)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Finished training...")


def load_dataset(data_dir, data_type, nl_tokenzier, pl_tokenizer, overwrite=False,
                 thread_num=None, num_limit=None, resample_rate=1):
    cache_dir = os.path.join(data_dir, "cache")
    if not thread_num:
        thread_num = multiprocessing.cpu_count()
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cached_file = os.path.join(cache_dir, "model3_cached_{}.dat".format(data_type))
    example_debug_file = os.path.join(cache_dir, "model3_debug_{}.dat".format(data_type))
    if os.path.exists(cached_file) and not overwrite:
        logger.info("Loading features from cached file %s", cached_file)
        dataset = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        csn_reader = CodeSearchNetReader(data_dir)
        examples = csn_reader.get_examples(type=data_type, num_limit=num_limit, summary_only=True)
        logger.info(
            "Creating features for {} dataset with num of {}".format(data_type, len(examples)))
        save_examples(examples, example_debug_file)
        if data_type == "valid":
            dataset = convert_example_to_retrival_task_dataset(examples, nl_tokenzier, pl_tokenizer, thread_num)
        elif data_type == "train":
            dataset = convert_example_to_triplet_dataset(examples, nl_tokenzier, pl_tokenizer, thread_num,
                                                         resample_rate)
        logger.info("Saving features into cached file {}".format(cached_file))
        torch.save(dataset, cached_file)
    return dataset


def train(args, train_dataset, valid_dataset, model):
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            optmz_path = os.path.join(args.model_path, "optimizer.pt")
            sched_path = os.path.join(args.model_path, "scheduler.pt")
            if os.path.isfile(optmz_path) and os.path.isfile(sched_path):
                optimizer.load_state_dict(torch.load(optmz_path))
                scheduler.load_state_dict(torch.load(sched_path))

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch")
    step_bar = tqdm(total=t_total, desc="Step progress")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "text_ids": batch[1],
                "pos_code_ids": batch[3],
                "neg_code_ids": batch[5]
            }
            outputs = model(**inputs)
            loss = outputs['loss']
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                step_bar.update(1)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(ckpt_output_dir):
                        os.makedirs(ckpt_output_dir)
                    save_check_point(model, ckpt_output_dir, args, optimizer, scheduler)
                    logger.info("Saving optimizer and scheduler states to %s", ckpt_output_dir)

                if args.valid_step > 0 and global_step % args.valid_step == 0:
                    if args.valid_num:
                        f1, success_rate = evaluate(args, valid_dataset, model, args.valid_num)
                        tb_writer.add_scalar("valid_f1", f1, global_step)
                        tb_writer.add_scalar("success_rate", success_rate, global_step)
    step_bar.close()
    logger.info("Save the trained model...")
    model_output = os.path.join(args.output_dir, "final_model")
    if not os.path.isdir(model_output):
        os.mkdir(model_output)
    save_check_point(model, model_output, args, optimizer, scheduler)
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, dataset, model, eval_num, prefix="", print_detail=True):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(dataset)
    eval_sampler = RandomSampler(dataset, replacement=True, num_samples=eval_num)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    if print_detail:
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

    res = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "text_ids": batch[0],
                "code_ids": batch[2],
            }
            label = batch[4]
            nl_id = batch[5]
            pl_id = batch[6]
            pred = model.get_sim_score(**inputs)
            for n, p, prd, lb in zip(nl_id.tolist(), pl_id.tolist(), pred, label.tolist()):
                res.append((n, p, prd[1], lb))
    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    max_f1, out_p, out_re, out_thre = best_accuracy(df, threshold_interval=1)
    success_rate = topN_RPF(df, 3)

    tqdm.write("evaluate F1={}".format(max_f1))
    tqdm.write("Success@3={}".format(success_rate))
    return max_f1, success_rate


def index_exmaple_vecs(examples, NL_tokenizer, PL_tokenizer, threads):
    threads = min(threads, os.cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            TBertProcessor().process_example,
            NL_tokenizer=NL_tokenizer,
            PL_tokenizer=PL_tokenizer,
            max_length=512
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                desc="convert examples to positive features"
            )
        )

    rel_index = defaultdict(set)
    NL_index = dict()  # find instance by id
    PL_index = dict()
    nl_cnt = 0
    pl_cnt = 0
    for f in tqdm(features, desc="assign ids to examples"):
        # assign id to the features
        nl_id = "{}".format(nl_cnt)
        pl_id = "{}".format(pl_cnt)
        f[0]['id'] = nl_id
        f[1]['id'] = pl_id
        NL_index[nl_id] = f[0]
        PL_index[pl_id] = f[1]
        rel_index[nl_id].add(pl_id)
        nl_cnt += 1
        pl_cnt += 1
    return NL_index, PL_index, rel_index


def convert_example_to_retrival_task_dataset(examples, NL_tokenizer, PL_tokenizer, threads=1):
    pos = []
    neg = []
    NL_index, PL_index, rel_index = index_exmaple_vecs(examples, NL_tokenizer, PL_tokenizer, threads)
    for nl_cnt, nl_id in enumerate(NL_index):
        for pl_id in PL_index:
            if pl_id in rel_index[nl_id]:
                pos.append((NL_index[nl_id], PL_index[pl_id], 1))
            else:
                neg.append((NL_index[nl_id], PL_index[pl_id], 0))
    dataset = TBertProcessor().features_to_data_set(pos + neg, True)
    return dataset


def convert_example_to_triplet_dataset(examples, NL_tokenizer, PL_tokenizer, threads=1, resample_rate=1):
    features = []
    NL_index, PL_index, rel_index = index_exmaple_vecs(examples, NL_tokenizer, PL_tokenizer, threads)
    for nl_id in tqdm(NL_index.keys()):
        pos_pl_ids = rel_index[nl_id]
        selected_ids = TBertProcessor().sample_negative_examples_ids(list(PL_index.keys()), pos_pl_ids, resample_rate)
        for pos_pl_id in pos_pl_ids:
            for neg_pl_id in selected_ids:
                features.append((NL_index[nl_id], PL_index[pos_pl_id], PL_index[neg_pl_id]))
    return convert_triplets_to_dataset(features)  # 9 columns


def convert_triplets_to_dataset(features):
    all_NL_ids = torch.tensor([int(f[0]['id']) for f in features], dtype=torch.long)
    all_NL_input_ids = torch.tensor([f[0]['input_ids'] for f in features], dtype=torch.long)

    all_pos_PL_ids = torch.tensor([int(f[1]['id']) for f in features], dtype=torch.long)
    all_pos_PL_input_ids = torch.tensor([f[1]['input_ids'] for f in features], dtype=torch.long)

    all_neg_PL_ids = torch.tensor([int(f[2]['id']) for f in features], dtype=torch.long)
    all_neg_PL_input_ids = torch.tensor([f[2]['input_ids'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_NL_ids, all_NL_input_ids,
                            all_pos_PL_ids, all_pos_PL_input_ids,
                            all_neg_PL_ids, all_neg_PL_input_ids)
    return dataset


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


if __name__ == "__main__":
    main()
