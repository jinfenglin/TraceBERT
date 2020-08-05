import logging
import os
import sys

sys.path.append("../..")

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from common.data_structures import Examples
from common.utils import load_check_point, save_check_point, write_tensor_board, evaluate_retrival, \
    format_triplet_batch_input

from code_search.twin.twin_train import get_train_args, init_train_env, load_examples, \
    get_optimizer_scheduler, log_train_info

logger = logging.getLogger(__name__)


def train(args, train_examples, valid_examples, model):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("../runs")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    example_num = len(train_examples)
    epoch_batch_num = example_num / args.train_batch_size

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (epoch_batch_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    log_train_info(args, example_num, t_total)
    args.global_step = 1
    args.epochs_trained = 0
    args.steps_trained_in_current_epoch = 0

    if args.model_path and os.path.exists(args.model_path):
        ckpt = load_check_point(model, args.model_path, optimizer, scheduler)
        model, optimizer, scheduler, args = ckpt["model"], ckpt['optimizer'], ckpt['scheduler'], ckpt['args']
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch {}, global step {}".format(args.epochs_trained, args.global_step))
    else:
        logger.info("Start a new training")

    skip_n_steps_in_epoch = args.steps_trained_in_current_epoch  # in case we resume training
    model.zero_grad()
    train_iterator = trange(args.epochs_trained, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    step_bar = tqdm(initial=args.epochs_trained, total=t_total, desc="Steps")

    for _ in train_iterator:
        params = (
            args, model, train_examples, valid_examples, optimizer, scheduler, tb_writer, step_bar,
            skip_n_steps_in_epoch)
        train_with_triplet_neg_sampling(*params)
        args.epochs_trained += 1
        skip_n_steps_in_epoch = 0
        if args.max_steps > 0 and args.global_step > args.max_steps:
            break

    model_output = os.path.join(args.output_dir, "final_model")
    save_check_point(model, model_output, args, optimizer, scheduler)
    step_bar.close()
    train_iterator.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()


def train_with_triplet_neg_sampling(args, model, train_examples: Examples, valid_examples: Examples, optimizer,
                                    scheduler, tb_writer, step_bar, skip_n_steps):
    tr_loss, tr_ac = 0, 0
    batch_size = args.per_gpu_train_batch_size
    if args.neg_sampling == "random":
        train_dataloader = train_examples.random_triplet_dataloader(batch_size=batch_size)
    elif args.neg_sampling == "online":
        train_dataloader = train_examples.online_neg_sampling_dataloader(batch_size=batch_size)
    else:
        raise Exception("{} neg_sampling is not recoginized...".format(args.neg_sampling))

    for step, batch in enumerate(train_dataloader):
        if skip_n_steps > 0:
            skip_n_steps -= 1
            continue
        if args.neg_sampling == "online":
            batch = train_examples.make_online_triplet_sampling_batch(batch, model)
        model.train()

        inputs = format_triplet_batch_input(batch, train_examples, model)
        outputs = model(**inputs)
        loss = outputs['loss']
        pos_sim = outputs['pos_sim']
        neg_sim = outputs['neg_sim']
        tr_ac += int(torch.sum((pos_sim > neg_sim).int()).item())

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
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
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
                ckpt_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(args.global_step))
                save_check_point(model, ckpt_output_dir, args, optimizer, scheduler)

            if args.valid_step > 0 and args.global_step % args.valid_step == 0:
                # step invoke validation
                valid_examples.update_embd(model)

                pk, best_f1, map = evaluate_retrival(model, valid_examples, args.per_gpu_eval_batch_size,
                                                     "evaluation/runtime_eval")
                tb_data = {
                    "precision@3": pk,
                    "best_f1": best_f1,
                    "MAP": map
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
        args.steps_trained_in_current_epoch += 1
        if args.max_steps > 0 and args.global_step > args.max_steps:
            break


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='I')
    valid_examples = load_examples(args.data_dir, data_type="valid", model=model, num_limit=args.valid_num,
                                   overwrite=args.overwrite)
    train_examples = load_examples(args.data_dir, data_type="train", model=model, num_limit=args.train_num,
                                   overwrite=args.overwrite)
    train(args, train_examples, valid_examples, model)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
