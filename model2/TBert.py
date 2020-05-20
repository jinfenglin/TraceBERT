import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from transformers import PreTrainedModel, BertConfig, AutoTokenizer, AutoModelWithLMHead, RobertaTokenizer, \
    RobertaForSequenceClassification, TextClassificationPipeline, AutoModel, BertTokenizer

# create directly applyable dataset examples in squad.py
# conducting training run_squad.py
# define a new classification header bert_modeling.py
from transformers.modeling_bert import BertOnlyNSPHead, BertLayer, BertForNextSentencePrediction, BertPooler, BertModel


class RelationClassifyHeader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relation_layer = BertLayer(config)
        self.regression_header = BertOnlyNSPHead(config)
        self.pooler = BertPooler(config)

    def forward(self, code_hidden, text_hidden, code_attention_mask, text_attention_mask):
        sequence_output = self.relation_layer(code_hidden, code_attention_mask, 1, text_hidden, text_attention_mask)
        pooled_output = self.pooler(sequence_output[0])
        seq_relationship_score = self.regression_header(pooled_output)
        return nn.Softmax(dim=-1)(seq_relationship_score)


class TBert(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        cbert_model = "huggingface/CodeBERTa-small-v1"
        nbert_model = "bert-base-uncased"

        self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model)
        self.cbert = AutoModel.from_pretrained(cbert_model)

        self.ntokenizer = AutoTokenizer.from_pretrained(nbert_model)
        self.nbert = AutoModel.from_pretrained(nbert_model)
        self.cls = RelationClassifyHeader(config)

    def forward(
            self,
            code_ids=None,
            code_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            relation_label=None):
        n_outputs = self.nbert(text_ids)
        c_outputs = self.cbert(code_ids)

        c_hidden = c_outputs[0]
        n_hidden = n_outputs[0]

        if code_attention_mask is None:
            code_attention_mask = torch.ones(code_ids.size(), device=code_ids.device)
        if text_attention_mask is None:
            text_attention_mask = torch.ones(text_ids.size(), device=text_ids.device)

        logits = self.cls(c_hidden, n_hidden, code_attention_mask, text_attention_mask)
        output_dict = {"logits": logits}
        if relation_label is not None:
            loss_fct = CrossEntropyLoss()
            # loss_fct = BCEWithLogitsLoss()
            rel_loss = loss_fct(logits.view(2, -1)[0], relation_label.view(-1))
            output_dict['loss'] = rel_loss
        return output_dict  # (rel_loss), rel_score

    def create_embd(self, str, tokenizer):
        return tokenizer.encode(str, return_tensors='pt', add_special_tokens=True)


# debug
if __name__ == "__main__":
    config = BertConfig()
    t_bert = TBert(config)
    t_bert.state_dict()['my_flag'] = '1024'
    torch.save(t_bert.state_dict(), "./output/t_bert_state.pt")
    recovered_model = TBert(config)
    recovered_model.load_state_dict(torch.load("./output/t_bert_state.pt"))
    print(recovered_model.state_dict()['my_flag'])

    # s1 = 'def prepare_inputs_for_generation(self, input_ids, **kwargs): return {"input_ids": input_ids}'
    # s2 = 'prepare inputs for generation'
    # config.is_decoder = True
    # c_input_ids = t_bert.create_embd(s1, t_bert.ctokneizer)
    # n_input_ids = t_bert.create_embd(s2, t_bert.ntokenizer)
    # recovered_model.eval()
    # res = recovered_model(code_ids=c_input_ids,
    #                       text_ids=n_input_ids,
    #                       )
    # print(res)
