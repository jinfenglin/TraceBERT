python exec_TBert_experiment.py \
--data_dir ./data/code_search_net/python \
--do_train \
--do_eval \
--output_dir ./output \
--model_path ./output \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--logging_steps 1 \
--save_steps 25 \
--gradient_accumulation_steps 16 \
--num_train_epochs 5 \
--learning_rate 4e-5 \
--ckpt_eval_num 600 \

