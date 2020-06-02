import argparse
import logging
import multiprocessing
import os
import random
import sys

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

sys.path.append("..")
from model2 import CodeSearchNetReader, TBertProcessor
from model2.TBert import TBert

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(data_dir, data_type, nl_tokenzier, pl_tokenizer, is_training, overwrite=False,
                            thread_num=None, num_limit=None):
    """
    Create data set for training and evaluation purpose. Save the formated dataset as cache
    :param args:
    :param data_type:
    :param tokenizer:
    :param evaluate:
    :param output_examples:
    :param num_limit the max number of instances read from the data file
    :return:
    """
    # Load data features from cache or dataset file
    if not thread_num:
        thread_num = multiprocessing.cpu_count()

    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cached_file = os.path.join(cache_dir, "cached_{}.dat".format(data_type))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_file) and not overwrite:
        logger.info("Loading features from cached file %s", cached_file)
        dataset = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        csn_reader = CodeSearchNetReader(data_dir)
        examples = csn_reader.get_examples(type=data_type, num_limit=num_limit)
        logger.info("Creating features for {} dataset with num of {}".format(data_type, len(examples)))
        dataset = TBertProcessor().convert_examples_to_dataset(examples, nl_tokenzier, pl_tokenizer,
                                                               is_training=is_training, threads=thread_num)
        logger.info("Saving features into cached file {}".format(cached_file))
        torch.save(dataset, cached_file)
    return dataset


def train(args, train_dataset, valid_dataset, model):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
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

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
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
    tr_ac, logging_ac = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "text_ids": batch[0],
                "text_attention_mask": batch[1],
                "code_ids": batch[2],
                "code_attention_mask": batch[3],
                "relation_label": batch[4]
            }
            outputs = model(**inputs)
            loss = outputs['loss']
            logit = outputs['logits']
            y_pred = logit.data.max(1)[1]
            tr_ac += y_pred.eq(batch[4]).long().sum().item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("accuracy",
                                         (tr_ac - logging_ac) / args.logging_steps / (
                                                 args.train_batch_size * args.gradient_accumulation_steps),
                                         global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    logging_ac = tr_ac

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(ckpt_output_dir):
                        os.makedirs(ckpt_output_dir)
                    save_check_point(model, ckpt_output_dir, args, optimizer, scheduler)
                    if args.ckpt_eval_num:
                        valid_accuracy = evaluate(args, valid_dataset, model, args.ckpt_eval_num)
                        tb_writer.add_scalar("valid_accuracy", valid_accuracy, global_step)
                    logger.info("Saving optimizer and scheduler states to %s", ckpt_output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    logger.info("Save the trained model...")
    model_output = os.path.join(args.output_dir, "final_model")
    if not os.path.isdir(model_output):
        os.mkdir(model_output)
    save_check_point(model, model_output, args, optimizer, scheduler)
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, dataset, model, eval_num, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(dataset)
    eval_sampler = RandomSampler(dataset, replacement=True, num_samples=eval_num)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_correct = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "text_ids": batch[0],
                "text_attention_mask": batch[1],
                "code_ids": batch[2],
                "code_attention_mask": batch[3],
            }
            label = batch[4]
            outputs = model(**inputs)
            logit = outputs['logits']
            y_pred = logit.data.max(1)[1]
            batch_correct = y_pred.eq(label).long().sum().item()
            num_correct += batch_correct
            tqdm.write(
                "pre:{},label:{},correct_num:{}".format(y_pred.data.tolist(), label.data.tolist(), batch_correct))
    accuracy = num_correct / len(dataset)
    return accuracy


def save_check_point(model, ckpt_dir, args, optimizer, scheduler):
    torch.save(model, os.path.join(ckpt_dir, 't_bert.pt'))
    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))


def evaluate_checkpoint(checkpoint, eval_dataset, args, eval_num):
    """

    :param checkpoint:  path to checkpiont directory
    :param args:
    :param num_limit: size of valid dataset that will attend evaluation
    :return:
    """
    model = torch.load(os.path.join(checkpoint, 't_bert.pt'))
    model.to(args.device)
    if not eval_num:
        eval_num = len(eval_dataset)
    result = evaluate(args, eval_dataset, model, eval_num)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./data", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument(
        "--model_path", default=None, type=str, required=True,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--ckpt_eval_num", type=int, default=100,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model = TBert(BertConfig())
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    valid_dataset = load_and_cache_examples(args.data_dir, "valid",
                                            model.ntokenizer, model.ctokneizer,
                                            is_training=True, num_limit=100, overwrite=args.overwrite)
    # Training
    if args.do_train:
        # 3 tensors (all_NL_input_ids, all_PL_input_ids, labels)
        train_dataset = load_and_cache_examples(args.data_dir, "train",
                                                model.ntokenizer, model.ctokneizer,
                                                is_training=True, num_limit=100, overwrite=args.overwrite)
        global_step, tr_loss = train(args, train_dataset, valid_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = []
            checkpoints.append(os.path.join(args.output_dir, "final_model"))
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_path)
            checkpoints = [args.model_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            result = evaluate_checkpoint(checkpoint, valid_dataset, args, None)
            results[checkpoint] = result

    logger.info("Results: {}".format(results))
    return results


if __name__ == "__main__":
    main()
