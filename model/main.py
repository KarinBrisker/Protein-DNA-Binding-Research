import argparse
import datetime
import os
import torch.optim as optim
import torch.onnx
import random
from tqdm import tqdm
from dictionary import Dictionary, ProteinsDataset, ProteinsDatasetClassification, Vec2score
from torch.utils.data import Dataset
from model import SiameseClassifier
import numpy as np
import time
import torch
import torch.nn as nn
import pandas as pd
import sklearn
# from sklearn.metrics import precision_score, recall_score, confusion_matrix
import logging
from torch.nn import DataParallel
import pickle


# Hyper-parameters
def parse_args():
    parser = argparse.ArgumentParser(description='DNA Model - siamese notwork of Decomposable-attention')
    parser.add_argument('--data_path', type=str, default='../DNA_data/dataframe_dataset.csv', help='data corpus')
    parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--wdecay', type=float, default=5e-5, help='weight decay applied to all weights')
    parser.add_argument('--hidden_size', type=float, default=128, help='hidden size')
    parser.add_argument('--beta', type=float, default=0.5, help='beta')
    parser.add_argument('--num_layers', type=float, default=2, help='num layers bi-lstm')
    parser.add_argument('--num_amino_acids', type=float, default=21, help='num amino-acids types in protein')
    parser.add_argument('--num_nucleotides', type=float, default=4, help='num nucleotides types in Dna')
    parser.add_argument('--mode', type=int, default=2, help='0 - train ranking, 1 - get vectors, 2 - get scores, 3 - jasfer salex')
    parser.add_argument('--embedding_dim', type=float, default=64,
                        help='embedding dim of each nucleotide and amino-acid')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='introduces a Dropout layer on the outputs of each LSTM layer except the last layer')
    args = parser.parse_args()
    return args


"""
this function loads 11 features for each amino acid in a given Protein.
this features are: MMS exposure, un-ordering etc.
    output: dict. <protein_name> : 200(padded protein seq) * 11(amino acid features)
"""


### TODO: NOTICE I SCALED THE FEATURES
def init_amino_acid_data():
    print('loading amino acids embeddings')
    embeddings = {}
    L =[]
    for p in tqdm(dict.proteins):
        path = '../DNA_data/amino_acid_data/' + p + '.txt'
        emb = []
        with open(path, 'r') as f:
            emb_amino_acids = f.readlines()
            for emb_amino in emb_amino_acids:
                e = np.fromstring(emb_amino.strip(), dtype=float, sep='\t')
                e[0] = e[0]/100
                L.append(e[0])
                e = e+[0.001] * 11
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
    prot2seq = {}
    seq2prot = {}
    print('get proteins data')
    for p in tqdm(proteins):
        path = '../DNA_data/data/' + p + '.txt'
        curr_p_df = pd.read_csv(path, sep="\t")
        curr_p_df.columns = ["d1", "d2", "score"]
        curr_p_df["dna"] = (curr_p_df["d1"] + curr_p_df["d2"]).apply(lambda x: dna2idx(x))
        protein = list(dict.amino_acids2idx[c] for c in list(dict.protein2seq[p]) if c in dict.amino_acids)
        protein += [dict.amino_acids2idx['#']] * (200 - len(protein))
        prot2seq[p] = str(protein)
        seq2prot[str(protein)] = p
        curr_p_df["protein"] = [torch.LongTensor(protein)] * curr_p_df.shape[0]
        curr_p_df["protein_name"] = [p] * curr_p_df.shape[0]
        curr_p_array = np.array(curr_p_df[["protein", "protein_name", "dna", "score"]].values)
        data.append(curr_p_array)
    output = np.concatenate(data, axis=0)
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
        curr_p_df["diff"] = abs(curr_p_df["score1"] - curr_p_df["score2"]) > 0.3
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
                                                  proteins[int(len(proteins) * .8):int(len(proteins) * .85)], proteins[
                                                                                                              int(len(
                                                                                                                  proteins) * .85):]
    logging.info('train proteins:\n')
    logging.info(train_proteins)
    logging.info('dev proteins:\n')
    logging.info(dev_proteins)
    logging.info('test proteins:\n')
    logging.info(test_proteins)
    return train_proteins, dev_proteins, test_proteins


def predict_on_jasfer_selex(args, device):
    model = SiameseClassifier(args, device).double().to(device)
    path = '../model/2019_12_11_14:34:43/epoch_82.pt'
    # path = '../model/2019_12_10_16:33:28/epoch_63.pt'
    model.load_state_dict(torch.load(path))
    proteins = os.listdir('../DNA_data/jasfer_selex/fasta_sites')
    print('loading amino acids embeddings')
    embeddings = {}
    with open('dna_options.txt', 'r') as f:
        options = f.readlines()[0].split('[')
        f.close()
    L =[]
    for p in tqdm(proteins):
        path = '../DNA_data/jasfer_selex/amino_acids/' + p
        emb = []
        with open(path, 'r') as f:
            emb_amino_acids = f.readlines()
            for emb_amino in emb_amino_acids:
                e = np.fromstring(emb_amino.strip(), dtype=float, sep='\t')
                e[0] = e[0]/100
                L.append(e[0])
                e = e+[0.001] * 11
                emb.append(torch.tensor(e))
            emb += [np.array(np.zeros(11))] * (200 - len(emb_amino_acids))
            embeddings[p] = torch.tensor(np.stack(emb, axis=0))

    data = []
    for p in tqdm(proteins):
        path = '../DNA_data/jasfer_selex/fasta_sites/' + p
        with open(path, 'r') as f:
            p_seq = f.readlines()[1].lower()
            if len(p_seq) >= 200:
                continue
            protein = list(dict.amino_acids2idx[c] for c in list(p_seq) if c in dict.amino_acids)
            protein += [dict.amino_acids2idx['#']] * (200 - len(protein))
            f.close()
        # path = '../DNA_data/jasfer_selex/sites/' + p
        # with open(path, 'r') as f:
        #     lines = f.readlines()
        #     dnas = [lines[i] for i in range(len(lines)) if i % 2 != 0]
        #     for dna in dnas:
        #         idx = []
        #         dna=list(dna.strip())
        #         dna_8 = dna
        #         # for i in range(len(dna)):
        #         #     if dna[i].isupper():
        #         #         idx.append(i)
        #         # if (idx[-1] - idx[0] + 1) < 8:
        #         #     # left = 8-(idx[-1]-idx[0])
        #         #     dna_8 = dna[idx[0]:idx[0] + 8]
        #         #     if len(dna_8)<8:
        #         #         dna_8 = dna[idx[-1] - 8: idx[-1]]
        #         # else:
        #         #     dna_8 = dna[idx[0]:idx[0] + 8]
        #         for i in range(len(dna_8)-1, -1, -1):
        #             if dna_8[i].lower() == 'a':
        #                 dna_8.append('t')
        #             elif dna_8[i].lower() == 't':
        #                 dna_8.append('a')
        #             elif dna_8[i].lower() == 'c':
        #                 dna_8.append('g')
        #             elif dna_8[i].lower() == 'g':
        #                 dna_8.append('c')
        #         dna1 = ''.join(dna_8).lower()
        #         print(dna1 + '\n')
        #         # dna = dna.strip().lower()
        #         dna_idx = pd.Series(dna1).apply(lambda x: dna2idx(x))
        #         data.append([p, torch.Tensor(protein), dna_idx.values[0], 0.5])
        #     f.close()

            for option in options[1:]:
                d= torch.tensor([int(s) for s in option[:-1].split(',')])
                data.append([p, torch.Tensor(protein).to(device).long(), d.to(device).long(), -0.5])
    df = pd.DataFrame.from_records(data, columns=['protein_name', 'protein', 'dna_idx', 'label'])



    scores = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(df), 2048)):
            if i + 2048 > len(df):
                break
            scores_per_couples = model.score_per_couple(torch.stack(list(df["protein"].iloc[i:i+2048].values)), torch.stack(list(df["dna_idx"].iloc[i:i+2048].values)), torch.stack([embeddings[x].to(device) for x in df["protein_name"].iloc[i:i+2048]]))
            scores += scores_per_couples.squeeze().tolist()
        scores_per_couples = model.score_per_couple(torch.stack(list(df["protein"].iloc[i:len(df)].values)), torch.stack(list(df["dna_idx"].iloc[i:len(df)].values)), torch.stack([embeddings[x].to(device) for x in df["protein_name"].iloc[i:len(df)]]))
        scores += scores_per_couples.squeeze().tolist()
    df["y_pred"] = scores
    df.to_pickle('jasfer_not_binds.pkl')
    # print(scores)
    return df, embeddings



"""
input: model
output: number of trainable parameters in the model
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
train function
"""


def train(args, model, train_loader, optimizer, params, criterion):
    epoch_start_time = time.time()
    model.train()
    count = 0
    all_predictions = []
    all_targets = []
    total_loss, correct = 0., 0
    for i, data in enumerate(tqdm(train_loader), 0):
        model.zero_grad()
        # get the inputs - protein, dna, label, amino_acids, dna2
        proteins, dnas1, labels, amino_acids, dnas2 = data
        prediction = model(proteins, dnas1, dnas2, amino_acids)
        correct += (prediction.round().int().squeeze() == labels.int()).tolist().count(1)
        count += len(prediction)
        loss = criterion(prediction.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        total_loss += loss.item()
        all_predictions += prediction.round().squeeze().int().tolist()
        all_targets += labels.int().tolist()
    logging.info(confusion_matrix(all_targets, all_predictions))
    logging.info('| time: {:5.2f}s | train loss {:5.8f} | train accuracy {:4.6f} | lr {:2.5f} | train precision {:5.4f}'
                 '| train recall {:5.4f}'.format((time.time() - epoch_start_time), total_loss * 1.0 / count,
                                                 100.0 * correct /
                                                 count, optimizer.param_groups[0]['lr'],
                                                 precision_score(all_targets, all_predictions) * 100, recall_score(
            all_targets, all_predictions) * 100))


"""
classification - from couple to score - need to find threshold for classification to binds/not-binds
input: model and data
output: all scores per couples of proteins and dna's
"""


def classification(model, data_loader):
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_proteins, all_dnas, names = [], [], [], [], []
        for i, data in enumerate(tqdm(data_loader), 0):
            # labels - binding score
            proteins, dnas, labels, amino_acids, protein_names = data
            all_proteins += proteins.tolist()
            names += protein_names
            all_dnas += dnas.tolist()
            scores_per_couples = model(proteins, dnas, amino_acids)
            all_predictions += scores_per_couples.squeeze().tolist()
            all_targets += labels.tolist()
    scores_df = pd.DataFrame(list(zip(all_proteins, names, all_dnas, all_targets, all_predictions)), columns=['protein', 'protein_names', 'dna',
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
        all_predictions, all_targets, all_proteins, all_dnas, names = [], [], [], [], []
        for i, data in enumerate(tqdm(data_loader), 0):
            # labels - binding score
            proteins, dnas, labels, amino_acids, protein_names = data
            all_proteins += proteins.tolist()
            names += protein_names
            all_dnas += dnas.tolist()
            vec_per_couples = model.vec_of_couple(proteins, dnas, amino_acids)
            all_predictions += vec_per_couples.squeeze().tolist()
            all_targets += labels.tolist()
    scores_df = pd.DataFrame(list(zip(all_proteins, names, all_dnas, all_targets, all_predictions)), columns=['protein', 'protein_names', 'dna',
                                                                                                       'y_true',
                                                                                                       'y_pred'])
    return scores_df

"""
test function
"""


def test(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        epoch_start_time = time.time()
        all_predictions = []
        all_targets = []
        total_loss = 0.
        correct = 0
        count = 0
        for i, data in enumerate(tqdm(test_loader), 0):
            proteins, dnas1, labels, amino_acids, dnas2 = data
            prediction = model(proteins, dnas1, dnas2, amino_acids)
            correct += (prediction.round().int().squeeze() == labels.int()).tolist().count(1)
            count += len(prediction)
            loss = criterion(prediction.squeeze(), labels)
            total_loss += loss.item()
            all_predictions += prediction.round().squeeze().int().tolist()
            all_targets += labels.int().tolist()
        logging.info(confusion_matrix(all_targets, all_predictions))
        logging.info('| time: {:5.2f}s | test loss {:5.8f} | test accuracy {:4.6f} | test precision {:5.4f}'
                     '| test recall {:5.4f}'.format((time.time() - epoch_start_time), total_loss * 1.0 / count,
                                                    100.0 * correct /
                                                    count, precision_score(all_targets, all_predictions) * 100,
                                                    recall_score(all_targets, all_predictions) * 100))


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


def train_loop(args, device, amino_acids_emb):
    file_name = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    logging.basicConfig(filename=file_name + '.txt', level=logging.DEBUG, filemode='w')
    args.save_dir = os.path.join('../model', file_name)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        logging.info("\t{}={}".format(attr.upper(), value))
    train_proteins, dev_proteins, test_proteins = init_dataset(random.sample(dict.proteins, len(dict.proteins)))
    # train_proteins, dev_proteins, test_proteins = dict.proteins[:2], dict.proteins[2:4],dict.proteins[4:5]

    train_data, dev_data, test_data = get_proteins_data(train_proteins), get_proteins_data(
        dev_proteins), get_proteins_data(test_proteins)

    model = SiameseClassifier(args, device).double().to(device)
    path = '../model/2019_12_10_16:33:28/epoch_21.pt'
    model.load_state_dict(torch.load(path))

    model = DataParallel(model, device_ids=[2, 0, 1, 3], output_device=2)  # run on all 4 gpu
    logging.info('create data loaders')
    dev_loader = create_dataset_loader(dev_data, amino_acids_emb, device, args)
    test_loader = create_dataset_loader(test_data, amino_acids_emb, device, args)
    logging.info(f'The model has {count_parameters(model):,} trainable parameters')
    criterion = nn.BCELoss().to(device)
    params_model = filter(lambda p: p.requires_grad, model.parameters())
    params = list(params_model) + list(criterion.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    logging.info('\n --------- training --------\n')
    for epoch in range(1, args.epochs):
        logging.info('\n\n### epoch: ' + str(epoch) + ' ###\n\n')
        train_data = get_proteins_data(train_proteins)
        train_loader = create_dataset_loader(train_data, amino_acids_emb, device, args)

        train(args, model, train_loader, optimizer, params, criterion)
        test(model, dev_loader, criterion)
        logging.info('-' * 89)
        with open(os.path.join(args.save_dir, 'epoch_' + str(epoch) + '.pt'), 'wb') as f:
            torch.save(model.module.state_dict(), f)
        if epoch % 7 == 0:
            logging.info('\n\non real test:')
            test(model, test_loader, criterion)
            logging.info('\n\nend results on real test\n\n')

    test(model, test_loader, criterion)


def get_vec_of_couples(args, device, amino_acids_emb):
    print('getting vectores of couples...')
    train_proteins, dev_proteins, test_proteins = init_dataset(random.sample(dict.proteins, len(dict.proteins)))
    train_data, dev_data, test_data = get_proteins_data_classification(train_proteins), get_proteins_data_classification(
        dev_proteins), get_proteins_data_classification(test_proteins)
    # file_name = 'from_ranking_to_classification'
    # logging.basicConfig(filename=file_name + '.txt', level=logging.DEBUG, filemode='w')
    # logging.getLogger().setLevel(logging.INFO)
    model = SiameseClassifier(args, device).double().to(device)
    path = '../model/2019_12_11_14:34:43/epoch_82.pt'
    model.load_state_dict(torch.load(path))

    # model = DataParallel(model, device_ids=[2, 0, 1, 3], output_device=2)  # run on all 4 gpu
    print('loaded model')

    data_dict = {'test': test_data, 'dev': dev_data, 'train1': train_data[:1000000],
                 'train2': train_data[1000000:2000000],
                 'train3': train_data[2000000:3000000], 'train4': train_data[3000000:]}
    for item in data_dict.items():
        print(item[0])
        loader = create_dataset_loader_classification(item[1], amino_acids_emb, device, args)
        test_vec_rep = get_vector_rep_of_protein_and_dna(model, loader)
        test_vec_rep.to_pickle(f'{item[0]}_vec_rep_of_couples__no_zeros_2019_12_11_14:34:43_epoch_82.pkl')



def get_scores_per_couples(args, device, amino_acids_emb):
    print('getting scores of couples...')
    # train_proteins, dev_proteins, test_proteins = dict.proteins[:35], dict.proteins[35:70],dict.proteins[70:]
    # train_data, dev_data, test_data = get_proteins_data_classification(train_proteins), get_proteins_data_classification(
    #     dev_proteins), get_proteins_data_classification(test_proteins)
    # file_name = 'from_ranking_to_classification'
    # logging.basicConfig(filename=file_name + '.txt', level=logging.DEBUG, filemode='w')
    # logging.getLogger().setLevel(logging.INFO)
    model = SiameseClassifier(args, device).double().to(device)
    path = '../model/2019_12_11_14:34:43/epoch_82.pt'
    model.load_state_dict(torch.load(path))
    # model = DataParallel(model, device_ids=[2, 0, 1, 3], output_device=2)  # run on all 4 gpu
    vecs = []
    for p in dict.proteins:
        protein = list(dict.amino_acids2idx[c] for c in list(dict.protein2seq[p]) if c in dict.amino_acids)
        vec_p = model.get_protein_vec(torch.tensor(protein).to(device).unsqueeze(0), amino_acids_emb[p][:len(protein)].to(device).unsqueeze(0))
        vecs.append([p, vec_p.tolist()])
    data=pd.DataFrame.from_records(vecs, columns=['protein_name', 'vec'])
    data.to_pickle('142_proteins_vectors.pkl')
    exit()
    print('woopi doopi')
    # model = SiameseClassifier(args, device).double().to(device)
    # path = '../model/2019_12_10_16:33:28/epoch_21.pt'
    # model.load_state_dict(torch.load(path))
    # loader = create_dataset_loader_classification(test_data, amino_acids_emb, device, args)
    # test_vec_rep = classification(model, loader)
    # test_vec_rep.to_pickle(f'test_data_score_of_couples__no_zeros_2019_12_10_16:33:28_epoch_21.pkl')
    # model = SiameseClassifier(args, device).double().to(device)
    # path = '../model/2019_12_11_14:34:43/epoch_82.pt'
    # model.load_state_dict(torch.load(path))
    # loader = create_dataset_loader_classification(test_data, amino_acids_emb, device, args)
    # test_vec_rep = classification(model, loader)
    # test_vec_rep.to_pickle(f'test_data_score_of_couples__no_zeros_2019_12_11_14:34:43_epoch_82.pkl')
    exit()
"""
main function
"""


def main():
    args = parse_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:2")
    amino_acids_emb = init_amino_acid_data()

    # 0 - train ranking
    if args.mode == 0:
        print('training...')
        train_loop(args, device, amino_acids_emb)

    # 1 - classification - inference time
    elif args.mode == 1:
        print('vectors...')
        get_vec_of_couples(args, device, amino_acids_emb)

    elif args.mode == 2:
        print('scores...')
        exit()
        get_scores_per_couples(args, device, amino_acids_emb)


    elif args.mode == 3:
        print('jasfer selex(...')
        predict_on_jasfer_selex(args, device)

    logging.info('finished')


if __name__ == '__main__':
    dict = Dictionary()
    main()


