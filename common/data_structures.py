from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.models import TBert, TwinBert
from common.utils import clean_space
from torch import Tensor

# keywords for features
F_ID = 'id'
F_TOKEN = 'toknes'
F_ATTEN_MASK = "attention_mask"
F_INPUT_ID = "input_ids"
F_EMBD = "embd"


class Examples:
    """
    Manage the examples read from raw dataset

    examples:
    valid_examples = CodeSearchNetReader(data_dir).get_examples(type="valid", num_limit=valid_num, summary_only=True)
    valid_examples = Examples(valid_examples)
    valid_examples.update_features(model)
    valid_examples.update_embd(model)

    """

    def __init__(self, raw_examples: List):
        self.NL_index, self.PL_index, self.rel_index = self.__index_exmaple(raw_examples)

    def __is_positive_case(self, nl_id, pl_id):
        if nl_id not in self.rel_index:
            return False
        rel_pls = set(self.rel_index[nl_id])
        return pl_id in rel_pls

    def __index_exmaple(self, raw_examples):
        """
        Raw examples should be a dictionary with key "NL" for natural langauge and PL for programming language.
        Each {NL, PL} pair in same dictionary will be regarded as related ones and used as positive examples.
        :param raw_examples:
        :return:
        """
        rel_index = defaultdict(set)
        NL_index = dict()  # find instance by id
        PL_index = dict()

        # hanlde duplicated NL and PL with reversed index
        reverse_NL_index = dict()
        reverse_PL_index = dict()

        nl_id_max = 0
        pl_id_max = 0
        for r_exp in tqdm(raw_examples, desc="assign ids to examples"):
            nl_tks = clean_space(r_exp["NL"])
            pl_tks = r_exp["PL"]

            if nl_tks in reverse_NL_index:
                nl_id = reverse_NL_index[nl_tks]
            else:
                nl_id = nl_id_max
                nl_id_max += 1

            if pl_tks in reverse_PL_index:
                pl_id = reverse_PL_index[pl_tks]
            else:
                pl_id = pl_id_max
                pl_id_max += 1

            NL_index[nl_id] = {F_TOKEN: nl_tks, F_ID: nl_id}
            PL_index[pl_id] = {F_TOKEN: pl_tks, F_ID: pl_id}  # keep space for PL
            rel_index[nl_id].add(pl_id)
            nl_id += 1
            pl_id += 1
        return NL_index, PL_index, rel_index

    def _gen_feature(self, example, tokenizer):
        feature = tokenizer.encode_plus(example[F_TOKEN], max_length=512,
                                        pad_to_max_length=True, return_attention_mask=True,
                                        return_token_type_ids=False)
        res = {
            F_ID: example[F_ID],
            F_INPUT_ID: feature[F_INPUT_ID],
            F_ATTEN_MASK: feature[F_ATTEN_MASK]}
        return res

    def __update_feature_for_index(self, index, tokenizer, n_thread):
        with Pool(n_thread) as p:
            worker = partial(self._gen_feature, tokenizer=tokenizer)
            features = list(tqdm(p.imap(worker, index.values(), chunksize=32), desc="Feature creation"))
            for f in features:
                id = f[F_ID]
                index[id][F_INPUT_ID] = f[F_INPUT_ID]
                index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]

    def update_features(self, model: TwinBert, n_thread=1):
        """
        Create or overwritten token_ids and attention_mask
        :param model:
        :return:
        """
        self.__update_feature_for_index(self.NL_index, model.get_nl_tokenizer(), n_thread)
        self.__update_feature_for_index(self.PL_index, model.get_pl_tokenizer(), n_thread)

    def __update_embd_for_index(self, index, sub_model):
        for id in tqdm(index, desc="creating embedding"):
            feature = index[id]
            input_tensor = torch.tensor(feature[F_INPUT_ID]).view(1, -1).to(sub_model.device)
            mask_tensor = torch.tensor(feature[F_ATTEN_MASK]).view(1, -1).to(sub_model.device)
            embd = sub_model(input_tensor, mask_tensor)[0]
            embd.to('cpu')
            index[id][F_EMBD] = embd

    def update_embd(self, model: TwinBert):
        """
        Create or overwritten the embedding
        :param model:
        :return:
        """
        with torch.no_grad():
            model.eval()
            self.__update_embd_for_index(self.NL_index, model.get_nl_sub_model())
            self.__update_embd_for_index(self.PL_index, model.get_pl_sub_model())

    def get_retrivial_task_dataloader(self, batch_size):
        """create retrivial task"""
        res = []
        for nl_id in self.NL_index:
            for pl_id in self.PL_index:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                res.append((nl_id, pl_id, label))
        dataset = DataLoader(res, batch_size=batch_size)
        return dataset

    def create_retrival_batch(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert a batch of (nl_id, pl_id,lable) into (nl_embd,pl_embd, label)"""
        nl_embds, pl_embds = [], []
        for nl_id, pl_id in zip(nl_id_tensor.tolist(), pl_id_tensor.tolist()):
            nl_embds.append(self.NL_index[nl_id][F_EMBD])
            pl_embds.append(self.PL_index[pl_id][F_EMBD])
        nl_tensor = torch.stack(nl_embds)
        pl_tensor = torch.stack(pl_embds)
        return nl_tensor, pl_tensor