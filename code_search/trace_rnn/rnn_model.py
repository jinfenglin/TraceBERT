import logging

import torch
from torch import nn, autograd
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def load_embd_from_file(embd_file_path):
    embd_matrix = []
    word2idx = {}
    idx = 0

    with open(embd_file_path, 'r', encoding='utf8') as fin:
        for line in tqdm(fin, "load embdding"):
            line = line.split()
            word = line[0]
            vec = [float(x) for x in line[1:]]
            embd_matrix.append(torch.tensor(vec, dtype=torch.float64))
            word2idx[word] = idx
            idx += 1
            # if idx>100:
            #     break

    embd_dim = len(embd_matrix[0])
    embd_matrix.append(torch.from_numpy(
        np.random.normal(scale=0.6, size=(embd_dim,))))
    embd_matrix = torch.stack(embd_matrix)
    word2idx['__UNK__'] = idx
    embd_num = len(embd_matrix)
    return {"embd_matrix": embd_matrix,
            "word2idx": word2idx,
            "embd_dim": embd_dim,
            "embd_num": embd_num}


def create_emb_layer(embd_info, trainable=True):
    embd_num = embd_info['embd_num']
    embd_dim = embd_info['embd_dim']
    embd_layer = nn.Embedding(embd_num, embd_dim)
    embd_layer.load_state_dict({'weight': embd_info['embd_matrix']})
    if not trainable:
        embd_layer.weight.requires_grad = False
    logger.info("finished building embedding layer")
    embd_info['embd_layer'] = embd_layer
    return embd_info


class classifyHeader(nn.Module):
    def __init__(self, hidden_size, rnn_type):
        super().__init__()
        bi = 1
        if rnn_type == 'bi_gru':
            bi = 2

        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(hidden_size * 4 * bi, hidden_size * 2 * bi)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_size * 2 * bi, 2)

    def forward(self, nl_hidden, pl_hidden):
        concated_hidden = torch.cat((nl_hidden, pl_hidden), 1)
        multi_hidden = torch.mul(nl_hidden, pl_hidden)
        diff_hidden = torch.abs(nl_hidden - nl_hidden)
        fuse_hidden = torch.cat((multi_hidden, diff_hidden), 1)
        fuse_hidden = torch.cat((fuse_hidden, concated_hidden), 1)
        fuse_hidden = self.dense(fuse_hidden)
        sigmoid_hidden = self.dropout(self.sigmoid(fuse_hidden))
        logits = self.output_layer(sigmoid_hidden)
        return logits


class RNNAvgPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


# class classifyHeader(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         # self.code_pooler = RNNAvgPooler(hidden_size)
#         # self.text_pooler = RNNAvgPooler(hidden_size)

#         # self.dense = nn.Linear(hidden_size * 3, hidden_size)
#         # self.dropout = nn.Dropout(0.2)
#         self.output_layer = nn.Linear(hidden_size * 3, 2)

#     def forward(self, nl_hidden, pl_hidden):
#         # pool_code_hidden = self.code_pooler(pl_hidden)
#         # pool_text_hidden = self.text_pooler(nl_hidden)

#         diff_hidden = torch.abs(nl_hidden - pl_hidden)
#         concated_hidden = torch.cat((nl_hidden, pl_hidden), 1)
#         concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

#         # x = self.dropout(concated_hidden)
#         # x = self.dense(x)
#         # x = torch.tanh(x)
#         # x = self.dropout(x)
#         x = self.output_layer(concated_hidden)
#         return x


class RNNEncoder(nn.Module):
    def __init__(self, hidden_dim, embd_info, max_seq_len, embd_trainable, is_no_padding, rnn_type):
        super().__init__()
        embd_info = create_emb_layer(embd_info, trainable=embd_trainable)
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.is_no_padding = is_no_padding
        self.max_seq_len = max_seq_len
        self.embedding = embd_info["embd_layer"]
        self.word2idx = embd_info['word2idx']
        self.embd_dim = embd_info['embd_dim']
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embd_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        elif rnn_type == 'bi_gru':
            self.rnn = nn.GRU(self.embd_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
                              bidirectional=True)

    def init_hidden(self, batch_size):
        if self.rnn_type == 'bi_gru':
            return torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(
                next(self.parameters()).device)
        else:
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
                    next(self.parameters()).device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
                    next(self.parameters()).device)
            )

    def token_to_ids(self, tokens):
        tokens = tokens[:self.max_seq_len]
        id_vec = []
        for tk in tokens:
            tk = tk.lower()
            tk = tk if tk in self.word2idx else "__UNK__"
            id = self.word2idx[tk]
            id_vec.append(id)
        pad_num = max(0, self.max_seq_len - len(id_vec))
        if not self.is_no_padding:
            id_vec = pad_num * [0] + id_vec
        id_tensor = torch.tensor(id_vec)
        return id_tensor

    def forward(self, input_ids):
        hidden = self.init_hidden(input_ids.size(0))
        embd = self.embedding(input_ids)
        output, (last_hidden, last_cell_state) = self.rnn(embd, hidden)
        return output[:, -1, :]


class RNNTracer(nn.Module):
    def __init__(self, hidden_dim, embd_info, embd_trainable, is_no_padding, max_seq_len=128, rnn_type='bi_gru'):
        super().__init__()
        self.device = None
        self.embd_info = embd_info
        self.nl_encoder = RNNEncoder(
            hidden_dim, self.embd_info, max_seq_len, embd_trainable, is_no_padding, rnn_type)
        self.pl_encoder = RNNEncoder(
            hidden_dim, self.embd_info, max_seq_len, embd_trainable, is_no_padding, rnn_type)
        self.cls = classifyHeader(hidden_dim, rnn_type)

    def forward(self, nl_hidden, pl_hidden, label=None):
        logits = self.cls(nl_hidden=nl_hidden, pl_hidden=pl_hidden)
        output_dict = {'logits': logits}
        if label is not None:
            loss_fct = CrossEntropyLoss()
            rel_loss = loss_fct(logits.view(-1, 2), label.view(-1))
            output_dict['loss'] = rel_loss
        return output_dict

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(nl_hidden=text_hidden, pl_hidden=code_hidden)
        sim_scores = torch.softmax(logits.view(-1, 2), 1).data.tolist()
        return [x[1] for x in sim_scores]

    def get_nl_hidden(self, nl_input):
        nl_hidden = self.nl_encoder(nl_input)
        return nl_hidden

    def get_pl_hidden(self, pl_input):
        pl_hidden = self.pl_encoder(pl_input)
        return pl_hidden


if __name__ == "__main__":
    embd_info = load_embd_from_file("./we/glove.6B.300d.txt")
    embd_info = create_emb_layer(embd_info)
    sent1 = ["this", "is", "sent1", "false"]
    sent2 = ["ok", "cool", "yes"]
    sent3 = ['that', "the", "an"]

    rt = RNNTracer(hidden_dim=100, embd_info=embd_info,
                   max_seq_len=128, embd_trainable=True)
    input_1 = rt.pl_encoder.token_to_ids(sent1)
    input_2 = rt.pl_encoder.token_to_ids(sent2)
    input_3 = rt.nl_encoder.token_to_ids(sent3)

    nl = input_1.view(1, -1)
    pl = input_2.view(1, -1)

    nl_hidden = rt.get_nl_hidden(nl)
    pl_hidden = rt.get_pl_hidden(pl)

    logits = rt(nl_hidden, pl_hidden)
    print(logits)

    score = rt.get_sim_score(text_hidden=nl_hidden, code_hidden=pl_hidden)
    print(score)
