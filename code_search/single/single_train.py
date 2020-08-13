import logging
import os
import sys

sys.path.append("../..")

import torch
from common.data_structures import Examples
from common.utils import save_check_point, format_batch_input_for_single_bert, \
    write_tensor_board, evaluate_classification, evalute_retrivial_for_single_bert

from code_search.twin.twin_train import get_train_args, init_train_env, load_examples, train

logger = logging.getLogger(__name__)


def train_single_iteration(args, model, train_examples: Examples, valid_examples: Examples, optimizer,
                           scheduler, tb_writer, step_bar, skip_n_steps):
    tr_loss, tr_ac = 0, 0
    batch_size = args.per_gpu_train_batch_size
    cache_file = "cached_single_random_neg_sample_epoch_{}.dat".format(args.epochs_trained)
    # save the examples for epoch
    if args.neg_sampling == "random":
        if args.overwrite or not os.path.isfile(cache_file):
            train_dataloader = train_examples.random_neg_sampling_dataloader(batch_size=batch_size)
            torch.save(train_dataloader, cache_file)
        else:
            train_dataloader = torch.load(cache_file)
    elif args.neg_sampling == "online":
        # we provide only positive cases and will create negative in the batch processing
        train_dataloader = train_examples.online_neg_sampling_dataloader(batch_size=int(batch_size / 2))
    else:
        raise Exception("{} neg_sampling is not recoginized...".format(args.neg_sampling))

    for step, batch in enumerate(train_dataloader):
        if skip_n_steps > 0:
            skip_n_steps -= 1
            continue
        if args.neg_sampling == "online":
            batch = train_examples.make_online_neg_sampling_batch(batch, model, args.hard_ratio)

        model.train()
        labels = batch[2].to(model.device)
        inputs = format_batch_input_for_single_bert(batch, train_examples, model)
        inputs['relation_label'] = labels
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
            if args.local_rank in [-1, 0] and args.save_steps > 0 and args.global_step % args.save_steps == 1:
                # step invoke checkpoint writing
                ckpt_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(args.global_step))
                save_check_point(model, ckpt_output_dir, args, optimizer, scheduler)

            if args.valid_step > 0 and args.global_step % args.valid_step == 1:
                # step invoke validation
                # valid_examples.update_embd(model)
                valid_accuracy, valid_loss = evaluate_classification(valid_examples, model,
                                                                     args.per_gpu_eval_batch_size,
                                                                     "evaluation/runtime_eval")
                pk, best_f1, map = evalute_retrivial_for_single_bert(model, valid_examples,
                                                                     args.per_gpu_eval_batch_size,
                                                                     "evaluation/runtime_eval")
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


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='single')
    valid_examples = load_examples(args.data_dir, data_type="valid", model=model, num_limit=args.valid_num,
                                   overwrite=args.overwrite)
    train_examples = load_examples(args.data_dir, data_type="train", model=model, num_limit=args.train_num,
                                   overwrite=args.overwrite)
    train(args, train_examples, valid_examples, model, train_single_iteration)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
