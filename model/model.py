import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal

torch.backends.cudnn.enabled = False


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
# https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/model_SentCNN.py
# https://github.com/nvnhat95/Natural-Language-Inference


class SelfAttention_Module(nn.Module):
    def __init__(self, hidden_dim, use_BN=True, dropout_rate=0.5):
        super(SelfAttention_Module, self).__init__()

        # linear -> BN -> relu -> dropout -> linear -> relu
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




class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout_rate):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout_rate)

    def initHiddenCell(self):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        return rand_hidden, rand_cell

    def forward(self, batch_size, input_data, hidden, cell):
        """ Performs a forward pass through the network. """
        output = self.embedding_table(input_data).view(1, batch_size, -1)
        for _ in range(self.opt.num_layers):
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        return output, hidden, cell


class SiameseModel(nn.Module):
    # hidden_dim = 128
    # embedding_dim = 128
    def __init__(self, use_bn=True, dropout_rate=0.4):
        super(SiameseModel, self).__init__()
        # embedding
        hidden_dim=128
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
        # self.encoder_a = self.encoder_b = LSTMEncoder(vocab_size, self.opt, is_train)

        # linear transformation from embedding
        self.input_fc = nn.Linear(self.embedding_dim, hidden_dim, bias=True)
        # linear transformation to prediction
        self.output_fc = nn.Linear(hidden_dim * 2, 1, bias=True)
        self.activation = nn.ReLU()
        self.features_linear_layer = nn.Linear(11, 100, bias=True)
        self.hidden = None
        self.hidden_dna = None

    def init_weights(self):
        initrange = 0.1
        self.embedding_matrix_p.weight.data.uniform_(-initrange, initrange)
        self.embedding_dna.weight.data.uniform_(-initrange, initrange)

    # TODO: INIT HIDDEN?? HOW TO INIT? WHAT IS THE OUTPUT?? I CALL IT TWICE WHAT HAPPENDS TO THE LOSS??
    def forward(self, p, d, amino_acids):
        # convert features using linear layer
        amino_acids = self.activation(self.features_linear_layer(amino_acids))

        # concat learnable embeddings to amino-acids features

        l_p = p.shape[1]
        l_d = d.shape[1]

        bs = p.shape[0]

        # protein to embeddings
        p = self.embedding_matrix_p(p)  # p: (batch_size x l_p x embedding_dim)
        # dna to embeddings
        d = self.embedding_dna(d)  # d: (batch_size x l_d1 x embedding_dim)

        p, (_, _) = self.LSTM(p, self.hidden)
        d, (_, _) = self.LSTM_dna(d)

        # second
        self.hidden = self.init_hidden(bs)
        self.hidden_dna = self.init_hidden_dna(bs)

        # add to proetin pre-defined features
        # first
        p1 = torch.cat([p, amino_acids.float()], dim=2)

        p = p1.contiguous().view(-1, l_p, self.hidden_dim * 2)
        d = d.contiguous().view(-1, l_d, self.hidden_dim * 2)

        h1 = self.feature_extractor_module(p, d)
        y_hat = self.output_fc(h1)

        # rank = self.activation(y_hat1 - y_hat2)

        return y_hat

    def init_hiddens(self, bsz):
        self.init_hidden(bsz)
        self.init_hidden_dna(bsz)

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
####################333



class LSTMEncoder(nn.Module):
    """ Implements the network type integrated within the Siamese RNN architecture. """
    def __init__(self, vocab_size, opt, is_train=False):
        super(LSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.opt = opt
        self.name = 'sim_encoder'

        # Layers
        self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.opt.embedding_dims,
                                            padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims, num_layers=1)

    def initialize_hidden_plus_cell(self, batch_size):
        """ Re-initializes the hidden state, cell state, and the forget gate bias of the network. """
        zero_hidden = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        zero_cell = Variable(torch.randn(1, batch_size, self.opt.hidden_dims))
        return zero_hidden, zero_cell

    def forward(self, batch_size, input_data, hidden, cell):
        """ Performs a forward pass through the network. """
        output = self.embedding_table(input_data).view(1, batch_size, -1)
        for _ in range(self.opt.num_layers):
            output, (hidden, cell) = self.lstm_rnn(output, (hidden, cell))
        return output, hidden, cell


import torch.optim as optim
class SiameseClassifier(nn.Module):
    """ Sentence similarity estimator implementing a siamese arcitecture. Uses pretrained word2vec embeddings.
    Different to the paper, the weights are untied, to avoid exploding/ vanishing gradients. """
    def __init__(self, vocab_size, pretrained_embeddings=None, is_train=False):
        super(SiameseClassifier, self).__init__()

        self.embedding_matrix_p = nn.Embedding(21, self.embedding_dim, padding_idx=20)
        self.embedding_dna = nn.Embedding(4, self.embedding_dim)

        # Initialize constituent network
        self.encoder_a = self.encoder_b = LSTMEncoder(vocab_size, is_train)
        self.encoder_dna1 = self.encoder_dna2 = LSTMEncoder(vocab_size, is_train)

        # Initialize network parameters
        self.initialize_parameters()
        # Declare loss function
        self.loss_function = nn.MSELoss()
        # Initialize network optimizers
        self.optimizer_a = optim.Adam(self.encoder_a.parameters(), lr=1e-3,
                                      betas=(0.5, 0.999))
        self.optimizer_b = optim.Adam(self.encoder_a.parameters(), lr=1e-3,
                                      betas=(0.5, 0.999))

    def forward(self):
        """ Performs a single forward pass through the siamese architecture. """
        # Checkpoint the encoder state
        state_dict = self.encoder_a.state_dict()

        # Obtain the input length (each batch consists of padded sentences)
        input_length = self.batch_a.size(0)

        # Obtain sentence encodings from each encoder
        hidden_a, cell_a = self.encoder_a.initialize_hidden_plus_cell(self.batch_size)
        for t_i in range(input_length):
            output_a, hidden_a, cell_a = self.encoder_a(self.batch_size, self.batch_a[t_i, :], hidden_a, cell_a)

        # Restore checkpoint to establish weight-sharing
        self.encoder_b.load_state_dict(state_dict)
        hidden_b, cell_b = self.encoder_b.initialize_hidden_plus_cell(self.batch_size)
        for t_j in range(input_length):
            output_b, hidden_b, cell_b = self.encoder_b(self.batch_size, self.batch_b[t_j, :], hidden_b, cell_b)

        # Format sentence encodings as 2D tensors
        self.encoding_a = hidden_a.squeeze()
        self.encoding_b = hidden_b.squeeze()

        # Obtain similarity score predictions by calculating the Manhattan distance between sentence encodings
        if self.batch_size == 1:
            self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1))
        else:
            self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))

    def get_loss(self):
        """ Calculates the MSE loss between the network predictions and the ground truth. """
        # Loss is the L1 norm of the difference between the obtained sentence encodings
        self.loss = self.loss_function(self.prediction, self.labels)


    def initialize_parameters(self):
        """ Initializes network parameters. """
        state_dict = self.encoder_a.state_dict()
        for key in state_dict.keys():
            if '.weight' in key:
                state_dict[key] = xavier_normal(state_dict[key])
            if '.bias' in key:
                bias_length = state_dict[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict[key][start:end].fill_(2.5)
        self.encoder_a.load_state_dict(state_dict)
        """ Initializes network parameters. """
        state_dict = self.encoder_dna1.state_dict()
        for key in state_dict.keys():
            if '.weight' in key:
                state_dict[key] = xavier_normal(state_dict[key])
            if '.bias' in key:
                bias_length = state_dict[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict[key][start:end].fill_(2.5)
        self.encoder_dna1.load_state_dict(state_dict)
