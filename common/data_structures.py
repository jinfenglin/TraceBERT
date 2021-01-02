import heapq
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from code_search.trace_rnn.rnn_model import RNNTracer
from common.models import TwinBert, TBertS
from torch import Tensor

from common.utils import format_batch_input, format_batch_input_for_single_bert, format_rnn_batch_input

F_ID = 'id'
F_TOKEN = 'tokens'
F_ATTEN_MASK = "attention_mask"
F_INPUT_ID = "input_ids"
F_EMBD = "embd"
F_TK_TYPE = "token_type_ids"

# epoch cache data name
CLASSIFY_RANDON_CACHE = 'classify_random_epoch_{}.cache'
CLASSIFY_NEG_SAMP_CACHE = 'classify_neg_epoch_{}.cache'
RETRIVE_RANDOM_CACHE = 'retrive_random_epoch_{}.cache'
RETRIVE_NEG_SAMP_CACHE = 'retrive_neg_epoch_{}.cache'


def exclude_and_sample(sample_pool, exclude, num):
    for id in exclude:
        sample_pool.remove(id)
    selected = random.sample(list(sample_pool), num)
    return selected


def sample_until_found(sample_pool, exclude, num, retry=3):
    cur_retry = retry
    res = []
    sample_pool = list(sample_pool)
    while num > 0 and retry > 0:
        cho = random.choice(sample_pool)
        if cho in exclude:
            cur_retry -= 1
            continue
        cur_retry = retry
        num -= 1
        res.append(cho)
    return res


def clean_space(text):
    return " ".join(text.split())


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

    def __len__(self):
        return len(self.rel_index)

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
        for r_exp in raw_examples:
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

    def _gen_seq_pair_feature(self, nl_id, pl_id, tokenizer):
        nl_tks = self.NL_index[nl_id][F_TOKEN]
        pl_tks = self.PL_index[pl_id][F_TOKEN]
        feature = tokenizer.encode_plus(
            text=nl_tks,
            text_pair=pl_tks,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            max_length=512,
            add_special_tokens=True
        )
        res = {
            F_INPUT_ID: feature[F_INPUT_ID],
            F_ATTEN_MASK: feature[F_ATTEN_MASK],
            F_TK_TYPE: feature[F_TK_TYPE]
        }
        return res

    def __update_feature_for_index(self, index, tokenizer, n_thread):
        for v in tqdm(index.values(), desc="update feature"):
            f = self._gen_feature(v, tokenizer)
            id = f[F_ID]
            index[id][F_INPUT_ID] = f[F_INPUT_ID]
            index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]
        # with Pool(n_thread) as p:
        # worker = partial(self._gen_feature, tokenizer=tokenizer)
        # features = list(tqdm(p.imap(worker, index.values(), chunksize=32), desc="update feature"))
        # for f in features:
        #     id = f[F_ID]
        #     index[id][F_INPUT_ID] = f[F_INPUT_ID]
        #     index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]

    def update_features(self, model: TwinBert, n_thread=1):
        """
        Create or overwritten token_ids and attention_mask
        :param model:
        :return:
        """
        self.__update_feature_for_index(self.PL_index, model.get_pl_tokenizer(), n_thread)
        self.__update_feature_for_index(self.NL_index, model.get_nl_tokenizer(), n_thread)

    def __update_embd_for_index(self, index, sub_model):
        for id in tqdm(index, desc="update embedding"):
            feature = index[id]
            input_tensor = torch.tensor(feature[F_INPUT_ID]).view(1, -1).to(sub_model.device)
            mask_tensor = torch.tensor(feature[F_ATTEN_MASK]).view(1, -1).to(sub_model.device)
            embd = sub_model(input_tensor, mask_tensor)[0]
            embd_cpu = embd.to('cpu')
            index[id][F_EMBD] = embd_cpu

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

    def get_chunked_retrivial_task_examples(self, chunk_query_num=-1, chunk_size=1000):
        """
        Cut the positive examples into chuncks. For EACH chunk generate queries at a size of query_num * chunk_size
        :param query_num: if query_num is -1 then create queries at a size of chunk_size * chunk_size
        :param chunk_size:
        :return:
        """
        rels = []
        for nid in self.rel_index:
            for pid in self.rel_index[nid]:
                rels.append((nid, pid))
        rel_dl = DataLoader(rels, batch_size=chunk_size)
        examples = []
        for batch in rel_dl:
            batch_query_idx = 0
            nids, pids = batch[0].tolist(), batch[1].tolist()
            for nid in nids:
                batch_query_idx += 1
                if chunk_query_num != -1 and batch_query_idx > chunk_query_num:
                    break
                for pid in pids:
                    label = 1 if self.__is_positive_case(nid, pid) else 0
                    examples.append((nid, pid, label))
        return examples

    def id_pair_to_embd_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_tensor = self._id_to_embd(nl_id_tensor, self.NL_index)
        pl_tensor = self._id_to_embd(pl_id_tensor, self.PL_index)
        return nl_tensor, pl_tensor

    def id_pair_to_feature_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_input_tensor, nl_att_tensor = self._id_to_feature(nl_id_tensor, self.NL_index)
        pl_input_tensor, pl_att_tensor = self._id_to_feature(pl_id_tensor, self.PL_index)
        return nl_input_tensor, nl_att_tensor, pl_input_tensor, pl_att_tensor

    def id_triplet_to_feature_triplet(self, nl_id_tensor, pos_pl_id_tensor, neg_pl_id_tensor):
        nl_input_tensor, nl_att_tensor = self._id_to_feature(nl_id_tensor, self.NL_index)
        pos_pl_input_tensor, pos_pl_att_tensor = self._id_to_feature(pos_pl_id_tensor, self.PL_index)
        neg_pl_input_tensor, neg_pl_att_tensor = self._id_to_feature(neg_pl_id_tensor, self.PL_index)
        return nl_input_tensor, nl_att_tensor, pos_pl_input_tensor, \
               pos_pl_att_tensor, neg_pl_input_tensor, neg_pl_att_tensor

    def _id_to_feature(self, id_tensor: Tensor, index):
        input_ids, att_masks = [], []
        for id in id_tensor.tolist():
            input_ids.append(torch.tensor(index[id][F_INPUT_ID]))
            if F_ATTEN_MASK in index[id]:
                att_masks.append(torch.tensor(index[id][F_ATTEN_MASK]))
        input_tensor = torch.stack(input_ids)
        att_tensor = None
        if att_masks:
            att_tensor = torch.stack(att_masks)
        return input_tensor, att_tensor

    def _id_to_embd(self, id_tensor: Tensor, index):
        embds = []
        for id in id_tensor.tolist():
            embds.append(index[id][F_EMBD])
        embd_tensor = torch.stack(embds)
        return embd_tensor

    def random_triplet_dataloader(self, batch_size):
        triplets = []
        for nl_id in tqdm(self.rel_index, desc="random_triplet_dataset"):
            pos_pl_ids = self.rel_index[nl_id]
            sample_num = len(pos_pl_ids)
            neg_pl_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for pos_id, neg_id in zip(pos_pl_ids, neg_pl_ids):
                triplets.append((nl_id, pos_id, neg_id))
        sampler = RandomSampler(triplets)
        dataset = DataLoader(triplets, batch_size=batch_size, sampler=sampler)
        return dataset

    def random_neg_sampling_dataloader(self, batch_size):
        pos, neg = [], []
        for nl_id in tqdm(self.rel_index, desc="random_neg_sampling_dataset"):
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((nl_id, n_id, 0))
        sampler = RandomSampler(pos + neg)
        dataset = DataLoader(pos + neg, batch_size=batch_size, sampler=sampler)
        return dataset

    def __rank_and_select(self, model: TwinBert, batch_size, top_n, search_space_percent=1.0):
        """
        rank the negative sampels based on the model socores and then select the top n.
        :param model:
        :param top_n:
        :param search_space_percent: generate a search space of len(pl_index) * len(nl_index) * search_space_percent.
        Use this parameter to control the
        :return:
        """
        res = []
        self.update_embd(model)  # use the updated embedding
        dataloader = self.get_retrivial_task_dataloader(batch_size)
        total_batch = len(dataloader) * search_space_percent + 1
        # TODO  remove code dupliation
        with tqdm(total=total_batch, desc="offline neg sampling rating") as sample_bar:
            for batch in dataloader:
                if total_batch <= 0:
                    break
                total_batch -= 1
                sample_bar.update()

                nl_ids = batch[0]
                pl_ids = batch[1]
                labels = batch[2]
                nl_embd, pl_embd = self.id_pair_to_embd_pair(nl_ids, pl_ids)
                model.eval()
                with torch.no_grad():
                    nl_embd = nl_embd.to(model.device)
                    pl_embd = pl_embd.to(model.device)

                    logits = model.cls(code_hidden=pl_embd, text_hidden=nl_embd)
                    pred = torch.softmax(logits, 1).data.tolist()
                    for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), pred, labels.tolist()):
                        if lb == 0:
                            res.append((n, p, lb, prd[1]))

        negs = heapq.nlargest(top_n, res, key=lambda x: x[3])  # find neg examples with largest pred score
        return negs

    def offline_neg_sampling_dataloader(self, model, batch_size):
        pos, neg = [], []
        for nl_id in self.rel_index:
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((nl_id, n_id, 0))
        pos_num = len(pos)
        hard_neg_num = int(pos_num * 0.9)  # number of hardest neg examples
        rand_neg_num = int(pos_num * 0.1)  # number of random neg examples
        hard_neg = self.__rank_and_select(model, batch_size, hard_neg_num, search_space_percent=0.001)
        neg = neg[:rand_neg_num] + hard_neg
        sampler = RandomSampler(pos + neg)
        dataset = DataLoader(pos + neg, batch_size=batch_size, sampler=sampler)
        return dataset

    def online_neg_sampling_dataloader(self, batch_size):
        pos = []
        for nl_id in self.rel_index:
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
        sampler = RandomSampler(pos)
        dataset = DataLoader(pos, batch_size=batch_size, sampler=sampler)
        return dataset

    def make_online_neg_sampling_batch(self, batch: Tuple, model: TwinBert, hard_ratio):
        """
        Convert positive examples into
        :param batch: tuples  containing the positive examples
        :return:
        """
        nl_ids = batch[0].tolist()
        pl_ids = batch[1].tolist()
        pos, neg = [], []
        cand_neg = []
        for nl_id in nl_ids:
            for pl_id in pl_ids:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                if label == 0:
                    cand_neg.append((nl_id, pl_id, 0))
                else:
                    pos.append((nl_id, pl_id, 1))
        neg_loader = DataLoader(cand_neg, batch_size=len(batch))
        neg_slots = len(pos)
        hard_neg_slots = max(1, int(hard_ratio * neg_slots))
        rand_neg_slots = max(0, neg_slots - hard_neg_slots)
        for neg_batch in neg_loader:
            with torch.no_grad():
                model.eval()
                if isinstance(model, TBertS):
                    inputs = format_batch_input_for_single_bert(neg_batch, self, model)
                elif isinstance(model, RNNTracer):
                    inputs = format_rnn_batch_input(neg_batch, self, model)
                else:
                    inputs = format_batch_input(neg_batch, self, model)
                outputs = model(**inputs)
                logits = outputs['logits']
                pred = torch.softmax(logits, 1).data.tolist()
                for nl, pl, score in zip(neg_batch[0].tolist(), neg_batch[1].tolist(), pred):
                    neg.append((nl, pl, score[1]))

        hard_negs = [(x[0], x[1], 0) for x in
                     heapq.nlargest(hard_neg_slots, neg, key=lambda x: x[2])]  # repalce score with label
        rand_negs = random.choices(cand_neg, k=rand_neg_slots)
        res = pos + hard_negs + rand_negs
        random.shuffle(res)
        r_nl, r_pl, r_label = [], [], []
        for r in res:
            r_nl.append(r[0])
            r_pl.append(r[1])
            r_label.append(r[2])
        return (torch.Tensor(r_nl), torch.Tensor(r_pl), torch.Tensor(r_label).long())

    def make_online_triplet_sampling_batch(self, batch: Tuple, model: TwinBert):
        nl_ids = batch[0].tolist()
        pl_ids = batch[1].tolist()
        neg = defaultdict(list)
        cand_neg = []
        res = []
        for nl_id in nl_ids:
            for pl_id in pl_ids:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                if label == 0:
                    cand_neg.append((nl_id, pl_id, 0))

        neg_loader = DataLoader(cand_neg, batch_size=len(batch))
        for neg_batch in neg_loader:
            with torch.no_grad():
                model.eval()
                inputs = format_batch_input(neg_batch, self, model)
                text_hidden = model.create_nl_embd(inputs['text_ids'], inputs['text_attention_mask'])[0]
                code_hidden = model.create_pl_embd(inputs['code_ids'], inputs['code_attention_mask'])[0]
                sim_scores = model.get_sim_score(text_hidden=text_hidden, code_hidden=code_hidden)
                # del text_hidden
                # del code_hidden
                for nl, pl, score in zip(neg_batch[0].tolist(), neg_batch[1].tolist(), sim_scores):
                    neg[nl].append((pl, score))

        for nl_id, pl_id in zip(nl_ids, pl_ids):
            if len(neg[nl_id]):
                hard_neg_exmp = heapq.nlargest(1, neg[nl_id], key=lambda x: x[1])[0]
                res.append((nl_id, pl_id, hard_neg_exmp[0]))

        r_nl, r_pos, r_neg = [], [], []
        for r in res:
            r_nl.append(r[0])
            r_pos.append(r[1])
            r_neg.append(r[2])
        return (torch.Tensor(r_nl), torch.Tensor(r_pos), torch.Tensor(r_neg).long())
