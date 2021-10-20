# TraceBERT

BERT based software traceability model for tracing the Natural Langauge (NL) artifacts to Programming Langauge (PL) artifacts. It is based on the CodeBert langauge model provided by Microsoft. Our approach contains two steps of training:
1. Code Search: Train the model on a large set of function-documentatoin pairs
2. Issue-Commit tracing: Continue the training from step 1 to fine-tune the model for issue-commit tracing. 

I  provide three types of models with different architectures:
- Siamese: Use one shared LM to encode both NL and PL artifacts 
- Single: Merge NL and PL into one sequence then use a single LM to create encode
- Twin: Use two seperate LM for NL and PL

The results shows Single Arch can achieve best performance, while the Siamese have relative lower accuracy and faster speed. 

This repo is for replication purpose, thus it only provide scripts for train and evalution, the prediction scripts for production use is not provided yet. I will work on it in next version.

## Installation
- Python >= 3.7 
- pytorch/1.1.0
- 1 GPU with CUDA 10.2 or 11.1

```
pip install -U pip setuptools 
pip install -r requirement.txt
```

## Step1:Code Search
Step 1 uses the code search dataset, which can be found in this [link](https://github.com/github/CodeSearchNet). 
It is also the dataset used for pre-training the CodeBert LM. 
I train model for python only where other langauges such as Java and Ruby are also available. 

### Train 
```
cd code_search/siamese2
python siamese2_train.py \
    --data_dir ../data/code_search_net/python \
    --output_dir ./output \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --logging_steps 10 \
    --save_steps 10000 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 8 \
    --learning_rate 4e-5 \
    --valid_num 200 \
    --valid_step 10000 \
    --neg_sampling random
```

### Evalaute 
```
cd code_search/siamese2
python siamese2_eval.py \
--data_dir ../data/code_search_net/python \
--model_path <model_path> \
--per_gpu_eval_batch_size 4 \
--exp_name "default exp name" \
```

## Step2:Issue-Commit tracing:
Step 2 uses the dataset collected from Github by myself, which can be found in this [link]( https://zenodo.org/record/4511291#.YB3tjyj0mbg)

### Train 
```
cd trace/trace_siamese
python train_trace_siamese.py \
    --data_dir ../data/git_data/dbcli/pgcli \
    --model_path <model_path> \ 
    --output_dir ./output \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --logging_steps 50 \
    --save_steps 1000 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 400 \
    --learning_rate 4e-5 \
    --valid_step 1000 \
    --neg_sampling online
```

### Evalaute 

```
python eval_trace_siamese.py \
    --data_dir ../data/git_data/pallets/flask \
    --model_path <model_path> \
    --per_gpu_eval_batch_size 4 \
    --exp_name "default exp name"

```

## Apply on your own data
You can replace the second step with your own tracing data, e.g. trace requirements to source code file. The easiest way to do this is formatting the data into the following csv schema please refer the data in step 2 for example. After formatting the data, you can use the train/eval scripts in step2 to conduct training and evaluatoin.

----

**commit_file:**

commit_id: unique id of the code artifact

diff: the actaul content of the code file in string, in our case is the code change set

summary: summary of the code file, will be merged with diff as a single string

commit_time: not used

files: not used

----

**issue_file:**

issue_id: unique id of the NL artifact

issue_desc: string of the content, will be merged with issue_comments

issue_comments: string of the content, will be merged with issue_desc

created_at: not used

closed_at: not used

---

**link_file:**

issue_id: ids from issue_file

commit_id: ids from commit_file

----

## Models
Single and Siamese Models from Step 2: https://drive.google.com/drive/folders/1nxJFg22zep9RtDMSw6N5VRCqIb5ALZwk?usp=sharing


## Citation

```
@inproceedings{lin2021traceability,
  title={Traceability transformed: Generating more accurate links with pre-trained BERT models},
  author={Lin, Jinfeng and Liu, Yalin and Zeng, Qingkai and Jiang, Meng and Cleland-Huang, Jane},
  booktitle={2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)},
  pages={324--335},
  year={2021},
  organization={IEEE}
}
```
