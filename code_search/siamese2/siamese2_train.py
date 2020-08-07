import sys

sys.path.append("../..")
from code_search.twin.twin_train import get_train_args, init_train_env, load_examples, train_with_neg_sampling, train, \
    logger


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='siamese2')
    valid_examples = load_examples(args.data_dir, data_type="valid", model=model, num_limit=args.valid_num,
                                   overwrite=args.overwrite)
    train_examples = load_examples(args.data_dir, data_type="train", model=model, num_limit=args.train_num,
                                   overwrite=args.overwrite)
    train(args, train_examples, valid_examples, model, train_iter_method=train_with_neg_sampling)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
