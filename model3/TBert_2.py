import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, BertConfig, AutoTokenizer, AutoModel
import torch.nn.functional as F


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class CosineTrainHeader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

    def similarity(self, vec1, vec2):
        return F.cosine_similarity(vec1, vec2)

    def forward(self, text_hidden, pos_code_hidden, neg_code_hidden):
        pool_pos_code_hidden = self.code_pooler(pos_code_hidden)
        pool_neg_code_hidden = self.code_pooler(neg_code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)

        anchor_sim = self.similarity(pool_text_hidden, pool_pos_code_hidden)
        neg_sim = self.similarity(pool_text_hidden, pool_neg_code_hidden)
        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss


class TBert2(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        cbert_model = "huggingface/CodeBERTa-small-v1"
        nbert_model = "huggingface/CodeBERTa-small-v1"

        self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model)
        self.cbert = AutoModel.from_pretrained(cbert_model)

        self.ntokenizer = AutoTokenizer.from_pretrained(nbert_model)
        self.nbert = AutoModel.from_pretrained(nbert_model)
        self.cls = CosineTrainHeader(config)

    def forward(
            self,
            text_ids=None,
            pos_code_ids=None,
            neg_code_ids=None,
    ):
        n_outputs = self.nbert(text_ids)
        c_pos_outputs = self.cbert(pos_code_ids)
        c_neg_outputs = self.cbert(neg_code_ids)

        n_hidden = n_outputs[0]
        c_pos_hidden = c_pos_outputs[0]
        c_neg_hidden = c_neg_outputs[0]

        loss = self.cls(n_hidden, c_pos_hidden, c_neg_hidden)
        output_dict = {"loss": loss}
        return output_dict

    def create_embd(self, str, tokenizer):
        return tokenizer.encode(str, return_tensors='pt', add_special_tokens=True)

    def get_sim_score(self, text_ids, code_ids):
        text_embed = self.nbert(text_ids)
        code_embed = self.cbert(code_ids)
        return F.cosine_similarity(text_embed, code_embed)


if __name__ == "__main__":
    pass
