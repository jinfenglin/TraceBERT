python3 train_trace_rnn.py \
    --data_dir ../data/code_search_net/python \
    --embd_file_path ./we/proj_embedding.txt \
    --output_dir ./output \
    --logging_steps 10 \
    --save_steps 10000 \
    --num_train_epochs 8 \
    --hidden_dim 60 \
    --max_seq_len 80 \
    --learning_rate 0.0001 \
    --learning_rate 5e-5 \
    --valid_num 200 \
    --valid_step 10000
    --is_no_padding