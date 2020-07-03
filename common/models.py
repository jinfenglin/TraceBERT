import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, PreTrainedModel


class TwinBert(PreTrainedModel):
    def get_nl_tokenizer(self):
        raise NotImplementedError

    def get_pl_tokenizer(self):
        raise NotImplementedError

    def create_nl_embd(self, input_ids, attention_mask):
        raise NotImplementedError

    def create_pl_embd(self, input_ids, attention_mask):
        raise NotImplementedError

    def get_nl_sub_model(self):
        raise NotImplementedError

    def get_pl_sub_model(self):
        raise NotImplementedError


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader2(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.output_layer = nn.Linear(config.hidden_size * 3, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)
        return self.output_layer(concated_hidden)


class TBert(TwinBert):
    def __init__(self, config):
        super().__init__(config)
        cbert_model = "huggingface/CodeBERTa-small-v1"
        # nbert_model = "bert-base-uncased"
        # nbert_model = "roberta-base"
        nbert_model = "huggingface/CodeBERTa-small-v1"

        self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model)
        self.cbert = AutoModel.from_pretrained(cbert_model)

        self.ntokenizer = AutoTokenizer.from_pretrained(nbert_model)
        self.nbert = AutoModel.from_pretrained(nbert_model)
        self.cls = RelationClassifyHeader2(config)

    def forward(
            self,
            code_ids=None,
            code_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            relation_label=None):
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]

        logits = self.cls(code_hidden=c_hidden, text_hidden=n_hidden)
        output_dict = {"logits": logits}
        if relation_label is not None:
            loss_fct = CrossEntropyLoss()
            rel_loss = loss_fct(logits.view(-1, 2), relation_label.view(-1))
            output_dict['loss'] = rel_loss
        return output_dict  # (rel_loss), rel_score

    def get_nl_tokenizer(self):
        return self.ntokenizer

    def get_pl_tokenizer(self):
        return self.ctokneizer

    def create_nl_embd(self, input_ids, attention_mask):
        return self.nbert(input_ids, attention_mask)

    def create_pl_embd(self, input_ids, attention_mask):
        return self.cbert(input_ids, attention_mask)

    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert
