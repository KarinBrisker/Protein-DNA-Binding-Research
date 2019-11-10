import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
import torch.nn as nn
import torch

torch.backends.cudnn.enabled = False

"""
# resources:

1) https://github.com/nvnhat95/Natural-Language-Inference  
    Decomposable-attention implementation for Natural language inference (NLI) - determining entailment and 
    contradiction relationships between a premise and a hypothesis
    
2) https://github.com/fionn-mac/Manhattan-LSTM/blob/master/PyTorch/manhattan_lstm.py
3) https://github.com/demelin/Sentence-similarity-classifier-for-pyTorch/blob/master/similarity_estimator/networks.py
    MaLSTM model for computing Semantic Similarity - Siamese Manhattan LSTM
    If 2 questions ave the same semantic meaning

4) https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_
network/main.py
    bi-LSTM
    
5) https://www.eggie5.com/130-learning-to-rank-siamese-network-pairwise-data
    learning to rank using siamese networks
"""


# CHECKED #

# F - bi-LSTM model
class DecomposableAttention(nn.Module):
    def __init__(self, hidden_dim, use_BN=True, dropout_rate=0.5):
        super(DecomposableAttention, self).__init__()

        # linear -> BN -> relu -> dropout -> linear -> relu
        def MLP(input_dim, output_dim, use_BN, dropout_rate):
            layers = [nn.Linear(input_dim, output_dim)]
            if use_BN:
                layers.append(nn.BatchNorm1d(output_dim, affine=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            return mlp

        self.hidden_dim = hidden_dim
        self.G = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)
        self.H = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)
        self.ReLU_activation = nn.ReLU()

    # CHECKED #
    # p - bs, p_len, embedding_size
    # d - bs, d_len, embedding_size
    def forward(self, p, d):
        l_a = p.shape[1]
        l_b = d.shape[1]

        # equation (1) in paper:
        e = torch.bmm(p, torch.transpose(d, 1, 2))  # e: (batch_size x l_a x l_b)

        # equation (2) in paper:
        beta = torch.bmm(F.softmax(e, dim=2), d)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), p)  # alpha: (batch_size x l_b x hidden_dim)

        # equation (3) in paper:
        a_cat_beta = torch.cat((p, beta), dim=2)  # (bs, l_a, hidden_dim*2)
        b_cat_alpha = torch.cat((d, alpha), dim=2)  # (bs, l_b, hidden_dim*2)

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

        # h - (bs, hidden_dim)
        return self.ReLU_activation(h)


class CNNText(nn.Module):

    def __init__(self, args, embedding_dim, kernel_num, device):
        super(CNNText, self).__init__()
        self.args = args

        D = embedding_dim
        Co = kernel_num
        Ks = args.kernel_sizes

        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, 128)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (BS, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = self.embed(x)  # (BS, W, D)

        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (BS, 1, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(BS, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(BS, Co), ...]*len(Ks)

        x = torch.cat(x, 1)  # (BS, kernel_num * 3)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # x = self.fc1(x)  # (N, C)
        return x


class BiLSTM(nn.Module):
    def __init__(self, args, input_size, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.ReLU_activation = nn.ReLU()
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to
        # form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
        # Default: 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=args.dropout, batch_first=True,
                            bidirectional=True)
        self.initialize_parameters()
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)  # 2 for bidirectional

    # x- (bs, protein_len, embedding_size)
    def forward(self, x):
        # Set initial states
        # sending tensors to the device of the input -> important for data parallelism
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).double().to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).double().to(x.device)

        # Forward propagate LSTM. out: tensor of shape (batch_size, seq_length, hidden_size * num directions)
        out, _ = self.lstm(x, (h0, c0))
        out = self.ReLU_activation(self.fc(out))

        # out- (bs, protein_len, hidden_size)
        return out

    def initialize_parameters(self):
        """ Initializes network parameters. """
        list_params = [self.lstm]
        for param in list_params:
            state_dict_p = param.state_dict()
            for key in state_dict_p.keys():
                if '.weight' in key:
                    try:
                        state_dict_p[key] = xavier_normal_(state_dict_p[key])
                    except:
                        continue
                if '.bias' in key:
                    bias_length = state_dict_p[key].size()[0]
                    start, end = bias_length // 4, bias_length // 2
                    state_dict_p[key][start:end].fill_(2.5)
            param.load_state_dict(state_dict_p)


class SiameseClassifier(nn.Module):
    """ Sentence similarity estimator implementing a siamese arcitecture. Uses pretrained word2vec embeddings.
    Different to the paper, the weights are untied, to avoid exploding/ vanishing gradients. """

    def __init__(self, args, device):
        super(SiameseClassifier, self).__init__()
        self.hidden_size = args.hidden_size
        self.ReLU_activation = nn.ReLU()
        self.Sigmoid_activation = torch.sigmoid
        # 11 -> 64
        self.features_linear_layer = nn.Linear(args.num_features, args.num_features_after_linear, bias=True)
        self.dense_protein = nn.Linear(args.num_features_after_linear + args.embedding_dim, 64, bias=True)
        self.args = args
        # protein - 128
        self.embedding_amino_acids = nn.Embedding(args.num_amino_acids, args.embedding_dim,
                                                  padding_idx=args.num_amino_acids - 1)
        # dna - 32
        self.embedding_nucleotides = nn.Embedding(args.num_nucleotides, args.embedding_dim_dna)
        # Initialize constituent network
        self.encoder_protein1 = CNNText(args, 64, 100, device)
        # Initialize constituent network
        self.encoder_dna1 = CNNText(args, args.embedding_dim_dna, 30, device)
        self.feature_extractor_module = DecomposableAttention(self.hidden_size)
        self.output_fc = nn.Linear(390, 1, bias=True)

        # Initialize network parameters
        self.initialize_parameters()
        self.device = device

    def forward(self, p, d1, d2, amino_acids):
        """ Performs a single forward pass through the siamese architecture. """
        # amino_acids- (bs, l_p, 11 features)   ->   (bs, l_p, 8)
        amino_acids = self.ReLU_activation(self.features_linear_layer(amino_acids))
        # p- (bs, l_p) -> (bs, l_p, embedding_size)
        p = self.embedding_amino_acids(p)
        # concat learnable embeddings to amino-acids features - (bs, l_p, embedding_size + 8)
        p = self.ReLU_activation(self.dense_protein(torch.cat([p, amino_acids], dim=2)))
        # Bi-LSTM. output for each amino-acid and than MLP+ReLU- (bs, l_p, embedding_size + 8)-> (bs, l_p, 20)
        p = self.encoder_protein1(p)
        # (bs, l_d)   ->   (bs, l_d, embedding_dim)
        d1 = self.embedding_nucleotides(d1)
        # (bs, l_d)   ->   (bs, l_d, embedding_dim)
        d2 = self.embedding_nucleotides(d2)

        #  Bi-LSTM. output for each amino-acid and than MLP+ReLU- (bs, l_d, embedding_dim) -> (bs, l_d, 128)
        d1 = self.encoder_dna1(d1)
        #  Bi-LSTM. output for each amino-acid and than MLP+ReLU- (bs, l_d, embedding_dim) -> (bs, l_d, 128)
        d2 = self.encoder_dna1(d2)

        h1 = torch.cat((p, d1), dim=1)  # bs, 390
        h2 = torch.cat((p, d2), dim=1)  # bs. 390
        # h1, h2 - (bs * hidden_dim)
        # h1 = self.feature_extractor_module(p, d1)
        # h2 = self.feature_extractor_module(p, d2)

        # y_hat - (bs * 1)  -> (-0.2) to (+0.2) - the score of the first and the second pair
        y_hat1 = self.output_fc(h1)
        y_hat2 = self.output_fc(h2)

        # sigmoid -> 0 to 1, if less than 0.5 means second is higher
        rank = self.Sigmoid_activation(y_hat1 - y_hat2)
        # print(rank)
        return rank

    def initialize_parameters(self):
        """ Initializes network parameters. """
        list_params = [self.encoder_protein1, self.encoder_dna1]
        for param in list_params:
            state_dict_p = param.state_dict()
            for key in state_dict_p.keys():
                if '.weight' in key:
                    try:
                        state_dict_p[key] = xavier_normal_(state_dict_p[key])
                    except:
                        continue
                if '.bias' in key:
                    bias_length = state_dict_p[key].size()[0]
                    start, end = bias_length // 4, bias_length // 2
                    state_dict_p[key][start:end].fill_(2.5)
            param.load_state_dict(state_dict_p)
