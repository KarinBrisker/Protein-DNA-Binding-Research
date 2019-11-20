import argparse
import datetime
import os
import torch.optim as optim
import torch.onnx
import random
from tqdm import tqdm
from dictionary import Dictionary, ProteinsDataset, ProteinsDatasetClassification
from torch.utils.data import Dataset
from model import SiameseClassifier
import numpy as np
import time
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import logging
from torch.nn import DataParallel


# Hyper-parameters
def parse_args():
    parser = argparse.ArgumentParser(description='DNA Model - siamese notwork of Decomposable-attention')
    parser.add_argument('--data_path', type=str, default='../DNA_data/dataframe_dataset.csv', help='data corpus')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--wdecay', type=float, default=5e-5, help='weight decay applied to all weights')
    parser.add_argument('--input_size', type=float, default=128, help='input size')
    parser.add_argument('--hidden_size', type=float, default=128, help='hidden size')
    parser.add_argument('--beta', type=float, default=0.5, help='beta')
    parser.add_argument('--num_layers', type=float, default=1, help='num layers bi-lstm')
    parser.add_argument('--num_amino_acids', type=float, default=21, help='num amino-acids types in protein')
    parser.add_argument('--num_nucleotides', type=float, default=4, help='num nucleotides types in Dna')
    parser.add_argument('--train_or_classification', type=int, default=1, help='1 - train ranking, 1 - classification')
    parser.add_argument('--embedding_dim', type=float, default=64,
                        help='embedding dim of each nucleotide and amino-acid')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='introduces a Dropout layer on the outputs of each LSTM layer except the last layer')
    args = parser.parse_args()
    return args


"""
this function loads 11 features for each amino acid in a given Protein.
this features are: MMS exposure, un-ordering etc.
    output: dict. <protein_name> : 200(padded protein seq) * 11(amino acid features)
"""


def init_amino_acid_data():
    print('loading amino acids embeddings')
    embeddings = {}
    for p in tqdm(dict.proteins):
        path = '../DNA_data/amino_acid_data/' + p + '.txt'
        emb = []
        with open(path, 'r') as f:
            emb_amino_acids = f.readlines()
            for emb_amino in emb_amino_acids:
                e = np.fromstring(emb_amino.strip(), dtype=float, sep='\t')
                emb.append(torch.tensor(e))
            emb += [np.array(np.zeros(11))] * (200 - len(emb_amino_acids))
            embeddings[p] = torch.tensor(np.stack(emb, axis=0))
    return embeddings


# converts dna to sequence indexes tensor
def dna2idx(x):
    dna_list = [dict.dna2idx[c] for c in list(x.lower()) if c in dict.dna_vocab]
    return torch.LongTensor(dna_list)


"""
input: list of proteins names
output: 3D numpy array of:    protein, dna, binding_score
"""


def get_proteins_data_classification(proteins):
    data = []
    print('get proteins data')
    for p in tqdm(proteins):
        path = '../DNA_data/data/' + p + '.txt'
        curr_p_df = pd.read_csv(path, sep="\t")
        curr_p_df.columns = ["d1", "d2", "score"]
        curr_p_df["dna"] = (curr_p_df["d1"] + curr_p_df["d2"]).apply(lambda x: dna2idx(x))
        protein = list(dict.amino_acids2idx[c] for c in list(dict.protein2seq[p]) if c in dict.amino_acids)
        protein += [dict.amino_acids2idx['#']] * (200 - len(protein))
        curr_p_df["protein"] = [torch.LongTensor(protein)] * curr_p_df.shape[0]
        curr_p_df["protein_name"] = [p] * curr_p_df.shape[0]
        curr_p_array = np.array(curr_p_df[["protein", "protein_name", "dna", "score"]].values)
        data.append(curr_p_array)
    output = np.concatenate(data, axis=0)
    np.random.shuffle(output)
    return output


"""
input: list of proteins names
output: 3D numpy array of:    protein, dna, binding_score
"""


def get_proteins_data(proteins):
    data = []
    print('get proteins data')
    for p in tqdm(proteins):
        path = '../DNA_data/data/' + p + '.txt'
        curr_p_df = pd.read_csv(path, sep="\t")
        curr_p_df.columns = ["d1", "d2", "score1"]
        curr_p_df["dna1"] = (curr_p_df["d1"] + curr_p_df["d2"]).apply(lambda x: dna2idx(x))
        protein = list(dict.amino_acids2idx[c] for c in list(dict.protein2seq[p]) if c in dict.amino_acids)
        protein += [dict.amino_acids2idx['#']] * (200 - len(protein))
        curr_p_df["protein"] = [torch.LongTensor(protein)] * curr_p_df.shape[0]
        curr_p_df["protein_name"] = [p] * curr_p_df.shape[0]
        second_dna = np.array(curr_p_df[["dna1", "score1"]].values)
        np.random.shuffle(second_dna)
        curr_p_df = pd.concat([curr_p_df, pd.DataFrame(second_dna, columns=['dna2', 'score2'])], axis=1)
        curr_p_df["diff"] = abs(curr_p_df["score1"] - curr_p_df["score2"]) > 0.2
        # remove rows if 2 dna's are close, and if it's the same dna(the first condition captures both)
        df = curr_p_df[curr_p_df["diff"] == True]
        curr_p_array = np.array(df[["protein", "protein_name", "dna1", "score1", "dna2", "score2"]].values)
        data.append(curr_p_array)
    output = np.concatenate(data, axis=0)
    return output


"""
input: list of proteins names
output: 3 - 3D numpy array of: protein, dna, binding_score for train, dev and test
"""


def init_dataset(proteins):
    print('loading train dev and test - split')
    train_proteins, dev_proteins, test_proteins = proteins[:int(len(proteins) * .8)], \
                        proteins[int(len(proteins) * .8):int(len(proteins) * .85)], proteins[int(len(proteins) * .85):]
    logging.info('train proteins:\n')
    logging.info(train_proteins)
    logging.info('dev proteins:\n')
    logging.info(dev_proteins)
    logging.info('test proteins:\n')
    logging.info(test_proteins)
    return train_proteins, dev_proteins, test_proteins


"""
input: model
output: number of trainable parameters in the model
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
train function
"""


def train(args, model, train_loader, optimizer, params, criterion, device):
    epoch_start_time = time.time()
    model.train()
    count = 0
    total_loss, correct = 0., 0
    for i, data in enumerate(tqdm(train_loader), 0):
        model.zero_grad()
        # get the inputs - protein, dna, label, amino_acids
        proteins, dnas, labels, amino_acids = data
        prediction = model(proteins, dnas, amino_acids)
        count += len(prediction)
        loss = criterion(prediction.squeeze(), labels.to(device).double())
        optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        total_loss += loss.item()
    logging.info('| time: {:5.2f}s | train loss {:5.8f}'.format((time.time() - epoch_start_time), total_loss * 1.0 / count))


"""
classification - from couple to score - need to find threshold for classification to binds/not-binds
input: model and data
output: all scores per couples of proteins and dna's
"""


def classification(model, data_loader):
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_proteins, all_dnas = [], [], [], []
        for i, data in enumerate(tqdm(data_loader), 0):
            # labels - binding score
            proteins, dnas, labels, amino_acids = data
            all_proteins += proteins.tolist()
            all_dnas += dnas.tolist()
            scores_per_couples = model.module.score_per_couple(proteins, dnas, amino_acids)
            all_predictions += scores_per_couples.squeeze().tolist()
            all_targets += labels.tolist()
    scores_df = pd.DataFrame(list(zip(all_proteins, all_dnas, all_targets, all_predictions)), columns=['protein', 'dna',
                                                                                                       'y_true',
                                                                                                       'y_pred'])
    return scores_df


"""
from couple to vec representation - using trained model "siamese decomposable-attention" to get this vector.
 
input: model and data
output: all vectors per couples of proteins and dna's
"""


def get_vector_rep_of_protein_and_dna(model, data_loader):
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_proteins, all_dnas = [], [], [], []
        for i, data in enumerate(tqdm(data_loader), 0):
            # labels - binding score
            proteins, dnas, labels, amino_acids = data
            all_proteins += proteins.tolist()
            all_dnas += dnas.tolist()
            vec_per_couples = model.module.vec_of_couple(proteins, dnas, amino_acids)
            all_predictions += vec_per_couples.squeeze().tolist()
            all_targets += labels.tolist()
    scores_df = pd.DataFrame(list(zip(all_proteins, all_dnas, all_targets, all_predictions)), columns=['protein', 'dna',
                                                                                                       'y_true',
                                                                                                       'y_pred'])
    return scores_df


"""
test function
"""


def test(model, test_loader, criterion, device):
    model.eval()
    all_predictions, all_targets, all_proteins, all_dnas = [], [], [], []

    with torch.no_grad():
        count = 0
        for i, data in enumerate(tqdm(test_loader), 0):
            proteins, dnas, labels, amino_acids = data
            prediction = model(proteins, dnas, amino_acids)
            count += len(prediction)
            all_predictions += prediction.squeeze().tolist()
            all_targets += labels.tolist()
        pred = np.array(all_predictions)
        true = np.array(all_targets)
        tp = list((pred > 0.2) & (true > 0.2)).count(True)
        fp = list((pred > 0.2) & (true < 0.1)).count(True)
        tn = list((pred < 0.1) & (true < 0.1)).count(True)
        fn = list((pred < 0.1) & (true > 0.2)).count(True)
        print('-----')
        try:
            acc = (tp + tn) / (tp + tn + fp + fn) * 100
            precision = tp / (tp + fp) * 100
            recall = tp / (tp + fn) * 100
            print(
                f'      precision: {tp} / {tp + fp} = {precision}\n      recall: {tp} / {tp + fn} = {recall}\n      accuracy: {tp + tn} / {tp + tn + fp + fn} = {acc}')
            print(tp)
            print(fp)
            print(tn)
            print(fn)
        except:
            print(tp)
            print(fp)
            print(tn)
            print(fn)
"""
create_dataset_loader
"""


def create_dataset_loader(data, amino_acids_emb, device, args):
    # "protein", "protein_name", "dna1", "score1", "dna2", "score2"]
    dataset = ProteinsDataset(data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], amino_acids_emb,
                              device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loader


"""
create_dataset loader classification
"""


def create_dataset_loader_classification(data, amino_acids_emb, device, args):
    # "protein", "protein_name", "dna", "score"]
    dataset = ProteinsDatasetClassification(data[:, 0], data[:, 1], data[:, 2], data[:, 3], amino_acids_emb, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loader


"""
main function
"""


def main():
    args = parse_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(1234)
    device = torch.device("cuda:2")
    amino_acids_emb = init_amino_acid_data()

    # 0 - train ranking
    if args.train_or_classification == 0:
        file_name = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        logging.basicConfig(filename=file_name + '.txt', level=logging.DEBUG, filemode='w')
        args.save_dir = os.path.join('../model', file_name)
        logging.getLogger().setLevel(logging.INFO)
        logging.info("\nParameters:")
        for attr, value in sorted(args.__dict__.items()):
            logging.info("\t{}={}".format(attr.upper(), value))

        train_proteins, dev_proteins, test_proteins = init_dataset(random.sample(dict.proteins, len(dict.proteins)))
        train_data, dev_data, test_data = get_proteins_data_classification(train_proteins), \
                                          get_proteins_data_classification(dev_proteins), \
                                          get_proteins_data_classification(test_proteins)

        model = SiameseClassifier(args, device).double().to(device)
        # path = '../model/2019_11_06_10:47:31/epoch_' + str(406) + '.pt'
        # model.load_state_dict(torch.load(path))

        model = DataParallel(model, device_ids=[2, 0, 1, 3], output_device=2)  # run on all 4 gpu
        print('create data loaders')

        train_loader = create_dataset_loader_classification(train_data, amino_acids_emb, device, args)
        dev_loader = create_dataset_loader_classification(dev_data, amino_acids_emb, device, args)
        test_loader = create_dataset_loader_classification(test_data, amino_acids_emb, device, args)
        logging.info(f'The model has {count_parameters(model):,} trainable parameters')
        criterion = nn.MSELoss().to(device)
        params_model = filter(lambda p: p.requires_grad, model.parameters())
        params = list(params_model) + list(criterion.parameters())
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        print('\n --------- training --------\n')
        for epoch in range(1, args.epochs):
            train(args, model, train_loader, optimizer, params, criterion, device)
            test(model, dev_loader, criterion, device)
            logging.info('-' * 89)
            with open(os.path.join(args.save_dir, 'epoch_' + str(epoch) + '.pt'), 'wb') as f:
                torch.save(model.module.state_dict(), f)
        test(model, test_loader, criterion, device)
    else:

        train_proteins, dev_proteins, test_proteins = init_dataset(random.sample(dict.proteins, len(dict.proteins)))
        train_data = get_proteins_data_classification(train_proteins)
        dev_data = get_proteins_data_classification(dev_proteins)
        test_data = get_proteins_data_classification(test_proteins)

        model = SiameseClassifier(args, device).double().to(device)
        path = '../model/2019_11_20_10:07:23/epoch_17.pt'
        model.load_state_dict(torch.load(path))

        model = DataParallel(model, device_ids=[2, 0, 1, 3], output_device=2)  # run on all 4 gpu
        print('create data loaders')

        train_loader = create_dataset_loader_classification(train_data, amino_acids_emb, device, args)
        dev_loader = create_dataset_loader_classification(dev_data, amino_acids_emb, device, args)
        test_loader = create_dataset_loader_classification(test_data, amino_acids_emb, device, args)

        logging.info(f'The model has {count_parameters(model):,} trainable parameters')
        criterion = nn.MSELoss().to(device)
        test(model, train_loader, criterion, device)
        test(model, dev_loader, criterion, device)
        test(model, test_loader, criterion, device)


if __name__ == '__main__':
    dict = Dictionary()
    main()
