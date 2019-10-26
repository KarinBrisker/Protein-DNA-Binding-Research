import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
# https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/model_SentCNN.py
# https://github.com/nvnhat95/Natural-Language-Inference


class SelfAttention_Module(nn.Module):
    def __init__(self, hidden_dim, use_BN=True, dropout_rate=0.5):
        super(SelfAttention_Module, self).__init__()

        def MLP(input_dim, output_dim, use_BN, dropout_rate):
            layers = []
            layers.append(nn.Linear(input_dim, output_dim))
            if use_BN:
                layers.append(nn.BatchNorm1d(output_dim, affine=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            return mlp

        self.hidden_dim = hidden_dim
        # self.F = MLP(hidden_dim, hidden_dim, use_BN, dropout_rate)
        # self.F_dna = MLP(hidden_dim, hidden_dim, use_BN, dropout_rate)
        self.G = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)
        self.H = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)

    def forward(self, p, d):
        l_a = p.shape[1]
        l_b = d.shape[1]

        # p = self.F(p.view(-1, self.hidden_dim))  # p: ((batch_size * l_a) x hidden_dim)
        # p = p.view(-1, l_a, self.hidden_dim * 2)  # p: (batch_size x l_a x hidden_dim)
        # d = self.F_dna(d.view(-1, self.hidden_dim))  # d: ((batch_size * l_b) x hidden_dim)
        # d = d.view(-1, l_b, self.hidden_dim * 2)  # d: (batch_size x l_b x hidden_dim)

        # equation (1) in paper:
        e = torch.bmm(p, torch.transpose(d, 1, 2))  # e: (batch_size x l_a x l_b)

        # equation (2) in paper:
        beta = torch.bmm(F.softmax(e, dim=2), d)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), p)  # alpha: (batch_size x l_b x hidden_dim)

        # equation (3) in paper:
        a_cat_beta = torch.cat((p, beta), dim=2)
        b_cat_alpha = torch.cat((d, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * self.hidden_dim))  # v1: ((batch_size * l_a) x hidden_dim)
        v2 = self.G(b_cat_alpha.view(-1, 2 * self.hidden_dim))  # v2: ((batch_size * l_b) x hidden_dim)

        # equation (4) in paper:
        v1 = torch.sum(v1.view(-1, l_a, self.hidden_dim), dim=1)  # v1: (batch_size x 1 x hidden_dim)
        v2 = torch.sum(v2.view(-1, l_b, self.hidden_dim), dim=1)  # v2: (batch_size x 1 x hidden_dim)

        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)

        # equation (5) in paper:
        v1_cat_v2 = torch.cat((v1, v2), dim=1)  # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.H(v1_cat_v2)

        return h


#  model = Model(embedding_matrix=wordvec_matrix, hidden_dim=args.hidden_dim, \
#  feature_extractor=args.model_type, dropout_rate=args.dropout_rate, POS_embedding=POS_embedding).to(device)
class Model(nn.Module):
    def __init__(self, hidden_dim, use_bn=True, dropout_rate=0.4):
        super(Model, self).__init__()
        # embedding
        self.embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.embedding_matrix_p = nn.Embedding(21, self.embedding_dim, padding_idx=20)
        self.embedding_dna = nn.Embedding(4, self.embedding_dim)
        self.init_weights()

        self.feature_extractor_module = SelfAttention_Module(hidden_dim * 2, use_bn, dropout_rate)
        # minus - will concat 12 pre-trained features of the protein amino acids
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim - 6, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=dropout_rate)
        self.LSTM_dna = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
                                bidirectional=True, dropout=dropout_rate)

        # linear transformation from embedding
        self.input_fc = nn.Linear(self.embedding_dim, hidden_dim, bias=True)
        # linear transformation to prediction
        self.output_fc = nn.Linear(hidden_dim * 2, 1, bias=True)
        self.activation = nn.Sigmoid()

        self.hidden = None
        self.hidden_dna = None

    def init_weights(self):
        initrange = 0.1
        self.embedding_matrix_p.weight.data.uniform_(-initrange, initrange)
        self.embedding_dna.weight.data.uniform_(-initrange, initrange)

    def forward(self, p1, p2, d1, d2, ac1, ac2):
        l_p1 = p1.shape[1]
        l_p2 = p2.shape[1]
        l_d1 = d1.shape[1]
        l_d2 = d2.shape[1]

        bs = p1.shape[0]

        # first
        p1 = self.embedding_matrix_p(p1)  # p: (batch_size x l_p x embedding_dim)
        d1 = self.embedding_dna(d1)  # d1: (batch_size x l_d1 x embedding_dim)

        # second
        p2 = self.embedding_matrix_p(p2)  # p: (batch_size x l_p x embedding_dim)
        d2 = self.embedding_dna(d2)  # d2: (batch_size x l_d1 x embedding_dim)

        # first
        self.hidden = self.init_hidden(bs)
        self.hidden_dna = self.init_hidden_dna(bs)

        p1, (_, _) = self.LSTM(p1, self.hidden)
        d1, (_, _) = self.LSTM_dna(d1)

        # second
        self.hidden = self.init_hidden(bs)
        self.hidden_dna = self.init_hidden_dna(bs)

        p2, (_, _) = self.LSTM(p2, self.hidden)
        d2, (_, _) = self.LSTM_dna(d2, self.hidden_dna)

        # add to proetin pre-defined features
        # first
        p1 = torch.cat([p1, ac1.float()], dim=2)
        p1 = p1.contiguous().view(-1, l_p1, self.hidden_dim * 2)
        d1 = d1.contiguous().view(-1, l_d1, self.hidden_dim * 2)

        p2 = torch.cat([p2, ac2.float()], dim=2)
        p2 = p2.contiguous().view(-1, l_p2, self.hidden_dim * 2)
        d2 = d2.contiguous().view(-1, l_d2, self.hidden_dim * 2)

        h1 = self.feature_extractor_module(p1, d1)
        y_hat1 = self.output_fc(h1)

        h2 = self.feature_extractor_module(p2, d2)
        y_hat2 = self.output_fc(h2)

        rank = self.activation(y_hat1 - y_hat2)

        return rank

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        # num layers = 2 * num directions, bsz, nhid
        return (weight.new_zeros(2, bsz, self.hidden_dim - 6).contiguous(),
                weight.new_zeros(2, bsz, self.hidden_dim - 6).contiguous())

    def init_hidden_dna(self, bsz):
        # num layers = 2 * num directions, bsz, nhid
        weight = next(self.parameters())
        return (weight.new_zeros(2, bsz, self.hidden_dim).contiguous(),
                weight.new_zeros(2, bsz, self.hidden_dim).contiguous())
