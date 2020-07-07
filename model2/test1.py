import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import BertConfig

from common.data_processing import CodeSearchNetReader
from common.data_structures import Examples
from common.models import TBert



def relation_list(examples):
    rel = examples['rel']
    res = []
    for nl_id in rel.keys():
        for pl_id in rel[nl_id]:
            res.append((nl_id, pl_id, 1))
    return res


def find_embed(index, id_tensor: Tensor):
    embds = []
    for id in id_tensor.tolist():
        embds.append(index[id])
    return embds


def eval_retrival_test():
    data_dir = "./data/code_search_net/python"
    res_file = "test_retrival.csv"
    model = TBert(BertConfig())
    model.to("cuda")
    valid_num = 100

    valid_examples = CodeSearchNetReader(data_dir).get_examples(type="valid", num_limit=valid_num, summary_only=True)
    valid_examples = Examples(valid_examples)
    valid_examples.update_features(model)
    valid_examples.update_embd(model)
    eval_retrival(model, valid_examples, 128, res_file)


def eval_retrival(model, eval_examples: Examples, batch_size, res_file):
    retrival_dataloader = eval_examples.get_retrivial_task_dataloader(batch_size)
    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        nl_ids = batch[0]
        pl_ids = batch[1]
        labels = batch[2]
        nl_embd, pl_embd = eval_examples.id_pair_to_embd_pair(nl_ids, pl_ids)
        model.eval()
        with torch.no_grad():
            nl_embd.to(model.device)
            pl_embd.to(model.device)

            logits = model.cls(code_hidden=pl_embd, text_hidden=nl_embd)
            pred = torch.softmax(logits, 1).data.tolist()
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), pred, labels.tolist()):
                res.append((n, p, prd[1], lb))
    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    df.to_csv(res_file)


if __name__ == "__main__":
    eval_retrival_test()
    print("finished")
    # valid_dataset = relation_list(valid_examples)
    # sampler = RandomSampler(valid_dataset, replacement=True)
    # dataloader = DataLoader(valid_dataset, sampler=sampler, batch_size=8)
    #
    # for batch in dataloader:
    #     nl_ids = batch[0]
    #     pl_ids = batch[1]
    #     label = batch[2]
    #
    #     nl_embd = find_embed(valid_examples["NL_index"], nl_ids)
    #     pl_embd = find_embed(valid_examples['PL_index'], pl_ids)
    #
    #     print(nl_embd)
