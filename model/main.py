# coding: utf-8
import argparse
import datetime
import os
import torch.optim as optim
import torch.onnx
import random
from tqdm import tqdm
from dictionary import Dictionary
from model import Model
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import DataParallel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

parser = argparse.ArgumentParser(description='DNA Model')
parser.add_argument('--data_path', type=str, default='../../DNA_data/dataframe_dataset.csv', help='data corpus')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--wdecay', type=float, default=5e-5, help='weight decay applied to all weights')
parser.add_argument('--save', type=str, default='dna_142_lr_0_001.pt', help='path to save the final cl_model')
parser.add_argument('--logging_output', type=str, default='', help='logging output file name')
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
args = parser.parse_args()
logging.basicConfig(filename=args.logging_output, level=logging.DEBUG, filemode='w')
args.save_dir = os.path.join('../../model/LARGE_model', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logging.getLogger().setLevel(logging.INFO)

logging.info("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    logging.info("\t{}={}".format(attr.upper(), value))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(1234)
device = torch.device("cuda:1")
# device = "cpu"
logging.info(device)


"""
this function loads 11 features for each amino acid in a given Protein.
this features are: 
    1- MMS exposure
    2-9 secondary structure of the amino-acid
    10- amino-acid disorder
    11- inference

    output: dictionary. <protein_name> : 200(padded protein seq) * 12(amino acid features)
"""
# TODO: why 12 and not 11*200???
# TODO: why 200 and not the sequence length???


def init_amino_acid_data():
    logging.info('loading amino acids embeddings')
    embeddings = {}
    for p in dictionary.proteins:
        path = '../../DNA_data/amino_acid_data/'+p+'.txt'
        emb = []
        with open(path, 'r') as f:
            emb_amino_acids = f.readlines()
            for emb_amino in emb_amino_acids:
                # + '\t0' - dummy, so we have even length
                e = np.fromstring(emb_amino.strip() + '\t0', dtype=float, sep='\t')
                emb.append(torch.tensor(e))
            for i in range(200 - len(emb_amino_acids)):
                # + '\t0' - dummy, so we have even length
                emb.append(np.array(np.zeros(12)))
            protein_data = [dictionary.amino_acids2idx[c] for c in list(dictionary.protein2seq[p]) if c in dictionary.amino_acids]
            protein_data += [dictionary.amino_acids2idx['#']] * (200 - len(protein_data))
            embeddings[p] = torch.tensor(np.stack(emb, axis=0))
    return embeddings


def dna2idx(x):
    dna_list = [dictionary.dna2idx[c] for c in list(x.lower()) if c in dictionary.dna_vocab]
    return torch.LongTensor(dna_list)


"""
input: list of proteins names
output: 3D numpy array of:    protein, dna, binding_score
"""


def get_proteins_data(proteins):
    data = []
    for p in proteins:
        path = '../../DNA_data/data/' + p + '.txt'
        curr_p_df = pd.read_csv(path, sep="\t")
        curr_p_df.columns = ["d1", "d2", "score"]
        curr_p_df["dna"] = (curr_p_df["d1"] + curr_p_df["d2"]).apply(lambda x: dna2idx(x))
        protein = list(dictionary.amino_acids2idx[c] for c in list(dictionary.protein2seq[p]) if c in dictionary.amino_acids)
        protein += [dictionary.amino_acids2idx['#']] * (200 - len(protein))
        curr_p_df["protein"] = [torch.LongTensor(protein)] * curr_p_df.shape[0]
        curr_p_array = np.array(curr_p_df[["protein", "dna", "score"]].values)
        data.append(curr_p_array)
    return np.concatenate(data, axis=0)


"""
input: list of proteins names
output: 3 - 3D numpy array of: protein, dna, binding_score for train, dev and test
"""


def init_dataset(proteins):
    logging.info('loading train dev and test - split')
    train_proteins, dev_proteins, test_proteins = proteins[:int(len(proteins) * .8)], \
                        proteins[int(len(proteins) * .8):int(len(proteins) * .85)], proteins[int(len(proteins) * .85):]
    return get_proteins_data(train_proteins), get_proteins_data(dev_proteins), get_proteins_data(test_proteins)


"""
input: model
output: number of trainable parameters in the model
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    dictionary = Dictionary()
    amino_acids_emb = init_amino_acid_data()
    train_data, dev_data, test_data = init_dataset(random.sample(dictionary.proteins, len(dictionary.proteins)))
    model = Model(128).to(device)

    logging.info('create train loader')
    train_set = protein_vectors_dataset(train_data[:, 0], train_data[:, 1], train_data[:, 2], amino_acids_emb)
    logging.info('train - finished step 1')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    logging.info('train - finished step 2')

    logging.info('create dev loader')
    dev_set = protein_vectors_dataset(dev_data[:, 0], dev_data[:, 1], dev_data[:, 2], amino_acids_emb)
    logging.info('dev - finished step 1')
    devloader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size)
    logging.info('dev - finished step 2')

    logging.info('create test loader')
    test_set = protein_vectors_dataset(test_data[:, 0], test_data[:, 1], test_data[:, 2], amino_acids_emb)
    logging.info('test - finished step 1')
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)
    logging.info('test - finished step 2')

    logging.info('         ---     finished    ---         ')

    # model = DataParallel(model, device_ids=[2, 1, 0, 3], output_device=2)  # run on all 4 gpu

    logging.info(f'The model has {count_parameters(model):,} trainable parameters')
    criterion = nn.BCELoss().to(device)
    params_model = filter(lambda p: p.requires_grad, model.parameters())
    params = list(params_model) + list(criterion.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)
    # Loop over epochs.
    logging.info('\n\n training!!      :-) \n')
    for epoch in range(1, args.epochs + 100):
        logging.info('epoch: ' + str(epoch))
        epoch_start_time = time.time()
        model.train()
        count = 0
        all_predictions = []
        all_targets = []
        total_loss, correct = 0., 0
        start_time = time.time()
        for i, data in enumerate(tqdm(trainloader), 0):
            model.zero_grad()

            # get the inputs
            proteins, dnas, targets, amino_acids, proteins2, dnas2, targets2, amino_acids2 = data
            bs = proteins.shape[0]

            # hidden = model.init_hidden(targets.shape[0])
            # /hidden_dna = model.init_hidden_dna(targets.shape[0])
            # output = model(proteins.to(device), dnas.long().to(device), hidden, hidden_dna)
            diff = abs(targets - targets2) > 0.2
            label = torch.tensor([1 if (targets[i] - targets2[i]) > 0 else 0 for i in range(len(diff))])
            batch = torch.cat([diff.view(-1,1).float(), proteins.float(), proteins2.float(), dnas.float(),
                               dnas2.float(), label.view(-1,1).float(), amino_acids.view(bs,    -1).float(), amino_acids2.view(bs, -1).float()], dim=1)

            batch = batch[(batch[:, 0] == 1).nonzero().squeeze(1)]
            proteins, proteins2, dnas, dnas2, labels = batch[:, 1:201].clone().detach().view(-1, 200).long().to(device),\
                        batch[:, 201:401].clone().detach().view(-1, 200).long().to(device), \
                        (batch[:, 401:417]).clone().detach().view(-1, 16).long().to(device), (batch[:, 417:433]).clone().detach().long().view(-1, 16).to(device),\
                        (batch[:, 433]).clone().detach().view(-1, 1).to(device)
            amino_acids_len = amino_acids.shape[1]
            bs = proteins.shape[0]
            amino_acids, amino_acids2 = (batch[:, 434: 434 + amino_acids_len * 12]).view(bs, amino_acids_len , 12).clone().detach().to(device), \
                                        (batch[:, 434 + amino_acids_len * 12 :]).view(bs, -1, 12).clone().detach().to(device)
            prediction = model(proteins, proteins2, dnas, dnas2, amino_acids, amino_acids2)

            # prediction = torch.argmax(prediction, 1)

            p = prediction.int().squeeze(1)
            t = labels.int().squeeze(1)
            correct += list(p == t).count(1)
            count += int(prediction.shape[0])
            loss = criterion(prediction.squeeze(1), t.float())
            # loss = criterion(prediction, Variable(labels))
            optimizer.zero_grad()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            total_loss += loss.detach()
            all_predictions += p.tolist()
            all_targets += t.tolist()
        logging.info('\n')
        logging.info(confusion_matrix(all_targets, all_predictions))
        logging.info(f'train precision: {precision_score(all_targets, all_predictions)}, train recall: {recall_score(all_targets, all_predictions)}')
        logging.info('-' * 89)
        logging.info(
            '| time: {:5.2f}s | train loss {:5.8f} | train accuracy {:4.6f} | lr {:2.5f}'.format(
                (time.time() - start_time), total_loss * 1.0 / count, 100.0 * correct / count,
                optimizer.param_groups[0]['lr']))

        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            total_loss = 0.
            correct = 0
            total = 0
            count = 0
            for i, data in enumerate(tqdm(devloader), 0):
                proteins, dnas, targets, amino_acids, proteins2, dnas2, targets2, amino_acids2 = data
                bs = proteins.shape[0]
                diff = abs(targets - targets2) > 0.2
                label = torch.tensor([1 if (targets[i] - targets2[i]) > 0 else 0 for i in range(len(diff))])
                batch = torch.cat([diff.view(-1, 1).float(), proteins.float(), proteins2.float(), dnas.float(),
                                   dnas2.float(), label.view(-1, 1).float(), amino_acids.view(bs, -1).float(),
                                   amino_acids2.view(bs, -1).float()], dim=1)
                batch = batch[(batch[:, 0] == 1).nonzero().squeeze(1)]
                proteins, proteins2, dnas, dnas2, labels = batch[:, 1:201].clone().detach().view(-1, 200).long().to(
                    device), batch[:, 201:401].clone().detach().view(-1, 200).long().to(device), \
                   (batch[:, 401:417]).clone().detach().view(-1, 16).long().to(device), (
                   batch[:, 417:433]).clone().detach().long().view(-1, 16).to(device), \
                   (batch[:, 433]).clone().detach().view(-1, 1).to(device)

                amino_acids_len = amino_acids.shape[1]
                bs = proteins.shape[0]
                amino_acids, amino_acids2 = (batch[:, 434: 434 + amino_acids_len * 12]).view(bs, amino_acids_len,
                     12).clone().detach().requires_grad_(True).to(device), (batch[:, 434 + amino_acids_len * 12:]).view(bs, -1,
                     12).clone().detach().requires_grad_(True).to(device)
                prediction = model(proteins, proteins2, dnas, dnas2, amino_acids, amino_acids2)
                p = prediction.int().squeeze(1)
                t = labels.int().squeeze(1)
                correct += list(p == t).count(1)
                count += int(prediction.shape[0])
                loss = criterion(prediction.squeeze(1), t.float())
                all_predictions += p.tolist()
                all_targets += t.tolist()
                total_loss += loss.detach()
            logging.info('\n')
            logging.info(confusion_matrix(all_targets, all_predictions))
            logging.info(f'dev precision: {precision_score(all_targets, all_predictions)}, dev recall: {recall_score(all_targets, all_predictions)}')
            logging.info('-' * 89)
            logging.info('| time: {:5.2f}s | dev loss {:5.8f} | dev accuracy {:4.6f} | lr {:2.5f}'.format(
                (time.time() - start_time), total_loss * 1.0 / count, 100.0 * correct / count,
                optimizer.param_groups[0]['lr']))

        # with open(os.path.join(args.save_dir, 'epoch_' + str(epoch) + '.pt'), 'wb') as f:
        #     torch.save(model.state_dict(), f)

    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        total_loss = 0.
        correct = 0
        total = 0
        count = 0
        for i, data in enumerate(tqdm(testloader), 0):
            proteins, dnas, targets, amino_acids, proteins2, dnas2, targets2, amino_acids2 = data
            bs = proteins.shape[0]
            diff = abs(targets - targets2) > 0.2
            label = torch.tensor([1 if (targets[i] - targets2[i]) > 0 else 0 for i in range(len(diff))])
            batch = torch.cat([diff.view(-1, 1).float(), proteins.float(), proteins2.float(), dnas.float(),
                               dnas2.float(), label.view(-1, 1).float(), amino_acids.view(bs, -1).float(),
                               amino_acids2.view(bs, -1).float()], dim=1)
            batch = batch[(batch[:, 0] == 1).nonzero().squeeze(1)]
            proteins, proteins2, dnas, dnas2, labels = batch[:, 1:201].clone().detach().view(-1, 200).long().to(
                device), batch[:, 201:401].clone().detach().view(-1, 200).long().to(device), \
               (batch[:, 401:417]).clone().detach().view(-1, 16).long().to(device), (
               batch[:, 417:433]).clone().detach().long().view(-1, 16).to(device), \
               (batch[:, 433]).clone().detach().view(-1, 1).to(device)

            amino_acids_len = amino_acids.shape[1]
            bs = proteins.shape[0]
            amino_acids, amino_acids2 = (batch[:, 434: 434 + amino_acids_len * 12]).view(bs, amino_acids_len,
                 12).clone().detach().requires_grad_(True).to(device), (batch[:, 434 + amino_acids_len * 12:]).view(bs, -1,
                 12).clone().detach().requires_grad_(True).to(device)
            prediction = model(proteins, proteins2, dnas, dnas2, amino_acids, amino_acids2)
            p = prediction.int().squeeze(1)
            t = labels.int().squeeze(1)
            correct += list(p == t).count(1)
            count += int(prediction.shape[0])
            loss = criterion(prediction.squeeze(1), t.float())
            all_predictions += p.tolist()
            all_targets += t.tolist()
            total_loss += loss.detach()
        logging.info('\n')
        logging.info(confusion_matrix(all_targets, all_predictions))
        logging.info(f'test precision: {precision_score(all_targets, all_predictions)}, test recall: {recall_score(all_targets, all_predictions)}')
        logging.info('-' * 89)
        logging.info('| time: {:5.2f}s | test loss {:5.8f} | test accuracy {:4.6f} | lr {:2.5f}'.format(
            (time.time() - start_time), total_loss * 1.0 / count, 100.0 * correct / count,
            optimizer.param_groups[0]['lr']))

