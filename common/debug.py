import os

import pandas as pd


def debug_dataset(tuple_list, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    cnt = 0
    file_name = "debug_data_{}.csv".format(cnt)
    output_path = os.path.join(output_dir, file_name)
    while os.path.isfile(output_path):
        cnt += 1
        file_name = "debug_data_{}.csv".format(cnt)
        output_path = os.path.join(output_dir, file_name)
    col_names = range(len(tuple_list))
    df = pd.DataFrame(tuple_list, col_names)
    df.to_csv(output_path)
