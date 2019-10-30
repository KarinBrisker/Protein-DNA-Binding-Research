import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal_

torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch
import torch.optim as optim
# >>> rnn = nn.LSTM(10, 20, 2)
# >>> input = torch.randn(5, 3, 10)
# >>> h0 = torch.randn(2, 3, 20)
# >>> c0 = torch.randn(2, 3, 20)
# >>> output, (hn, cn) = rnn(input, (h0, c0))


# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
# bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
# dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
# bidirectional – If True, becomes a bidirectional LSTM. Default: False


# Hyper-parameters
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.utils import clip_grad_norm

sequence_length = 28
input_size = 128
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003
beta_1 = 0.5
num_amino_acids = 21
num_nucleotides = 4
# https://github.com/nvnhat95/Natural-Language-Inference
# https://github.com/fionn-mac/Manhattan-LSTM/blob/master/PyTorch/manhattan_lstm.py

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

        # equation (1) in paper:
        e = torch.bmm(p, torch.transpose(d, 1, 2))  # e: (batch_size x l_a x l_b)

        # equation (2) in paper:
        beta = torch.bmm(F.softmax(e, dim=2), d)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), p)  # alpha: (batch_size x l_b x hidden_dim)

        # equation (3) in paper:
        a_cat_beta = torch.cat((p, beta), dim=2)
        b_cat_alpha = torch.cat((d, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * self.hidden_dim).double())  # v1: ((batch_size * l_a) x hidden_dim)
        v2 = self.G(b_cat_alpha.view(-1, 2 * self.hidden_dim).double())  # v2: ((batch_size * l_b) x hidden_dim)

        # equation (4) in paper:
        v1 = torch.sum(v1.view(-1, l_a, self.hidden_dim), dim=1)  # v1: (batch_size x 1 x hidden_dim)
        v2 = torch.sum(v2.view(-1, l_b, self.hidden_dim), dim=1)  # v2: (batch_size x 1 x hidden_dim)

        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)

        # equation (5) in paper:
        v1_cat_v2 = torch.cat((v1, v2), dim=1)  # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.H(v1_cat_v2)

        return h

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
# https://github.com/demelin/Sentence-similarity-classifier-for-pyTorch/blob/master/similarity_estimator/networks.py
# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, drop_prob=0.5):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True, bidirectional=True).float().to(device)
        self.fc = nn.Linear(hidden_size * 2, num_classes).float()  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).float().to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).float().to(self.device)

        # Forward propagate LSTM
        x = x.float()
        out, _ = self.lstm.float()(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
#
        # Decode the hidden state of the last time step
        # out = self.fc.float()(out[:, -1, :])
        return out


# model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)


class SiameseClassifier(nn.Module):
    """ Sentence similarity estimator implementing a siamese arcitecture. Uses pretrained word2vec embeddings.
    Different to the paper, the weights are untied, to avoid exploding/ vanishing gradients. """
    def __init__(self, device):
        super(SiameseClassifier, self).__init__()
        self.activation = nn.ReLU()
        self.embedding_dim = 64
        self.features_linear_layer = nn.Linear(11, 64, bias=True)
        # protein
        self.embedding_amino_acids = nn.Embedding(num_amino_acids, self.embedding_dim, padding_idx=num_amino_acids - 1)
        # dna
        self.embedding_nucleotides = nn.Embedding(num_nucleotides, self.embedding_dim)
        # Initialize constituent network
        self.encoder_protein1 = self.encoder_protein2 = BiLSTM(input_size, hidden_size, num_layers, num_classes, device).float().to(device)
        # Initialize constituent network
        self.encoder_dna1 = self.encoder_dna2 = BiLSTM(self.embedding_dim, hidden_size, num_layers, num_classes, device).float().to(device)
        # Initialize network parameters
        self.initialize_parameters()
        # Declare loss function
        self.loss_function = nn.MSELoss()
        # Initialize network optimizers
        self.optimizer_a = optim.Adam(self.encoder_protein1.parameters(), lr=learning_rate,
                                      betas=(beta_1, 0.999))
        self.optimizer_b = optim.Adam(self.encoder_protein1.parameters(), lr=learning_rate,
                                      betas=(beta_1, 0.999))
        self.feature_extractor_module = SelfAttention_Module(hidden_size * 2)
        self.device = device
        self.output_fc = nn.Linear(hidden_size * 2, 1, bias=True)

    def forward(self, p1, p2, d1, d2, amino_acids1, amino_acids2):
        """ Performs a single forward pass through the siamese architecture. """
        amino_acids1 = self.activation(self.features_linear_layer(amino_acids1))
        amino_acids2 = self.activation(self.features_linear_layer(amino_acids2))

        p1 = self.embedding_amino_acids(p1)  # p: (batch_size x l_p x embedding_dim)
        p2 = self.embedding_amino_acids(p2)  # p: (batch_size x l_p x embedding_dim)
        d1 = self.embedding_nucleotides(d1)
        d2 = self.embedding_nucleotides(d2)

        # concat learnable embeddings to amino-acids features
        p1 = torch.cat([p1, amino_acids1], dim=2)
        p2 = torch.cat([p2, amino_acids2], dim=2)

        # Checkpoint the encoder state
        state_dict_protein = self.encoder_protein1.state_dict()
        state_dict_dna = self.encoder_dna1.state_dict()

        # Obtain sentence encodings from each encoder
        output_p1 = self.encoder_protein1(p1)
        output_d1 = self.encoder_dna1(d1)

        # Restore checkpoint to establish weight-sharing
        self.encoder_protein2.load_state_dict(state_dict_protein)
        self.encoder_dna2.load_state_dict(state_dict_dna)

        output_p2 = self.encoder_protein2(p2)
        output_d2 = self.encoder_dna2(d2)

        p1 = output_p1.contiguous().view(-1, p1.size(1), hidden_size * 2)
        d1 = output_d1.contiguous().view(-1, d1.size(1), hidden_size * 2)

        h1 = self.feature_extractor_module(p1, d1)

        p2 = output_p2.contiguous().view(-1, p1.size(1), hidden_size * 2)
        d2 = output_d2.contiguous().view(-1, d1.size(1), hidden_size * 2)

        h2 = self.feature_extractor_module(p2, d2)

        y_hat1 = self.output_fc(h1)
        y_hat2 = self.output_fc(h2)
        rank = self.activation(y_hat1 - y_hat2)

        return rank
        # # Obtain similarity score predictions by calculating the Manhattan distance between sentence encodings
        # if self.batch_size == 1:
        #     self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1))
        # else:
        #     self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))

    def get_loss(self):
        """ Calculates the MSE loss between the network predictions and the ground truth. """
        # Loss is the L1 norm of the difference between the obtained sentence encodings
        self.loss = self.loss_function(self.prediction, self.labels)

    def initialize_parameters(self):
        """ Initializes network parameters. """
        state_dict_p = self.encoder_protein1.state_dict()
        for key in state_dict_p.keys():
            if '.weight' in key:
                state_dict_p[key] = xavier_normal_(state_dict_p[key])
            if '.bias' in key:
                bias_length = state_dict_p[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict_p[key][start:end].fill_(2.5)
        self.encoder_protein1.load_state_dict(state_dict_p)
        state_dict_d = self.encoder_dna1.state_dict()
        for key in state_dict_d.keys():
            if '.weight' in key:
                state_dict_d[key] = xavier_normal_(state_dict_d[key])
            if '.bias' in key:
                bias_length = state_dict_d[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict_d[key][start:end].fill_(2.5)
        self.encoder_dna1.load_state_dict(state_dict_d)
# -- train --
# Get batches
# self.batch_a = train_batch_a
# self.batch_b = train_batch_b
# self.labels = train_labels
#
# # Get batch_size for current batch
# self.batch_size = self.batch_a.size(1)
#
# # Get gradients
# self.forward()
# self.encoder_protein1.zero_grad()  # encoder_protein1 == encoder_protein2
# self.get_loss()
# self.loss.backward()
#
# # Clip gradients
# clip_grad_norm(self.encoder_protein1.parameters(), self.opt.clip_value)
#
# # Optimize
# self.optimizer_a.step()


