python3 train_trace_rnn.py \
    --data_dir ../data/git_data/pallets/flask/ \
    --output_dir ./output \
    --embd_file_path ./we/proj_embedding.txt \
    --exp_name test_rnn \
    --valid_step 1000 \
    --logging_steps 10 \
    --learning_rate 0.001 \
    --num_train_epochs 100
