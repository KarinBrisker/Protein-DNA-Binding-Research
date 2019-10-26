# coding: utf-8
import argparse
import os
import torch.onnx
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from dictionary import Dictionary
dictionary = Dictionary()


'''
below 0.2 -> NO CONNECTION -> label=0
above 0.3 -> CONNECTED -> label=1
between 0.2 to 0.3 -> label None. I removed them
output : protein | DNA | label
'''


def files_to_labels(dir_path):
    proteins, DNAs, all_labels = [[] for i in range(3)]
    lines = open('protein_seq_2.txt').readlines()
    proteins_dict = {}
    for line in tqdm(lines): proteins_dict[line.split()[0]] = line.split()[1].lower()
    files_names = os.listdir('../../model/LARGE_model/DNA_data_2')
    for file in tqdm(files_names):
        protein = proteins_dict[file.split('.')[0]]
        lines = open(os.path.join(dir_path, file)).readlines()
        data = ([line.split()[0] + line.split()[1] for line in lines])
        labels = (
        [0 if float(line.split()[2]) < 0.2 else 1 if float(line.split()[2]) > 0.3 else np.nan for line in lines])
        proteins += [protein] * len(labels)
        all_labels += labels
        DNAs += data
    new_table = pd.DataFrame({'protein': proteins, 'dna': DNAs, 'label': all_labels}).dropna()
    new_table.to_csv('LARGE_dataframe_dataset.csv', encoding='utf-8', index=False)


def split_data_by_class():
    data = (pd.read_csv('../../model/LARGE_model/LARGE_dataframe_dataset.csv')).values
    binds_data = [i for i in data if i[2] == 1]
    not_binds_data = [i for i in data if i[2] == 0]
    np.save('not_binds_dataset.npy', not_binds_data)
    np.save('binds_dataset.npy', binds_data)
    binds_data = np.load('binds_dataset.npy')
    not_binds_data = np.load('not_binds_dataset.npy')
    return binds_data, not_binds_data

'''
shuffling all data and split it to train dev test
each contains even samples of binds and not binds
'''


def train_dev_test_split(binds_data, not_binds_data):
    np.random.shuffle(binds_data)
    split_1 = int(0.8 * len(binds_data))
    split_2 = int(0.9 * len(binds_data))
    train_data_binds = binds_data[:split_1]
    dev_data_1 = binds_data[split_1:split_2]
    test_data_1 = binds_data[split_2:]

    np.random.shuffle(not_binds_data)

    dev_data_2 = not_binds_data[:len(dev_data_1)]
    test_data_2 = not_binds_data[len(dev_data_1):len(dev_data_1) + len(test_data_1)]
    train_data_not_binds = not_binds_data[len(dev_data_1) + len(test_data_1):]

    dev_data = np.concatenate((dev_data_2, dev_data_1))
    np.random.shuffle(dev_data)

    test_data = np.concatenate((test_data_2, test_data_1))
    test_data = np.concatenate((test_data, train_data_not_binds[len(train_data_binds):]))
    np.random.shuffle(test_data)

    np.random.shuffle(train_data_not_binds)
    train_data = np.concatenate((train_data_not_binds[:len(train_data_binds)], train_data_binds))
    np.random.shuffle(train_data)

    return train_data, dev_data, test_data


def count_trigrams(train_proteins, test_proteins):
    dict_trigrams = {}
    for i in range(len(dictionary.trigrams)):
        dict_trigrams[dictionary.trigrams[i]] = 0
    # dict_trigrams = {dictionary.trigrams[int(i)]:0 for i in dictionary.trigrams}

    for protein in train_proteins:
        protein = protein.strip()
        protein_trigrams = [(protein[k:k + 3]) for k in range(len(list(protein)) - 3 + 1)]
        for tri in protein_trigrams:
            try:
                dict_trigrams[tri] += 1
            except:
                continue
                # print(tri)
    print(' --------------- ')
    for protein in test_proteins:
        protein = protein.strip()
        protein_trigrams = [(protein[k:k + 3]) for k in range(len(list(protein)) - 3 + 1)]
        for tri in protein_trigrams:
            if dict_trigrams[tri] == 0:
                print(tri)

    return dict_trigrams


'''
shuffling all data and split it to train dev test
each contains even samples of binds and not binds
* IF * protein is in the train set, then won't occur in test
'''


def train_dev_test_split_NO_OVERLAPPING():
    # basic data
    binds_data1 = np.load('/home/nlp/dahanka1/maliciousJS/DNA/model/basic_model/binds_dataset.npy')
    not_binds_data1 = np.load('/home/nlp/dahanka1/maliciousJS/DNA/model/basic_model/not_binds_dataset.npy')

    ## extra data
    binds_data2 = np.load('binds_dataset.npy')
    not_binds_data2 = np.load('not_binds_dataset.npy')
    binds_data = np.concatenate((binds_data1, binds_data2))
    not_binds_data = np.concatenate((not_binds_data1, not_binds_data2))
    np.random.shuffle(binds_data)
    count = 0

    train_proteins = []
    test_proteins = []

    train_data_binds = []
    test_data_1 = []

    train_data_not_binds = []
    test_data_2 = []
    lines = open(
        '/home/nlp/dahanka1/maliciousJS/DNA/model/LARGE_model/corpus/proteins_seq/protein_seq.txt').readlines()
    all_proteins = [line.strip().split('\t')[1].lower() for line in lines]
    for i in tqdm(range(len(binds_data))):
        # if len(binds_data[i][0]) > 1000:
        #     continue
        binds_data[i][1] = binds_data[i][1].lower()
        curr_p = binds_data[i][0]
        if curr_p not in all_proteins:
            continue
        if curr_p not in train_proteins and curr_p not in test_proteins:
            # not seen protein
            count += 1
            if count % 5 == 0:
                test_proteins.append(curr_p)
                test_data_1.append(binds_data[i])
            else:
                train_proteins.append(curr_p)
                train_data_binds.append(binds_data[i])
        else:
            # seen protein
            if curr_p in train_proteins:
                train_data_binds.append(binds_data[i])
            else:
                test_data_1.append(binds_data[i])

    dev_size = round(len(train_data_binds) * 0.1)

    for i in tqdm(range(len(not_binds_data))):
        not_binds_data[i][1] = not_binds_data[i][1].lower()
        curr_p = not_binds_data[i][0]
        if curr_p not in all_proteins:
            continue
        if curr_p in test_proteins:
            test_data_2.append(not_binds_data[i])
        else:
            train_data_not_binds.append(not_binds_data[i])

    np.random.shuffle(train_data_not_binds)
    np.random.shuffle(test_data_2)

    dev_data_1 = test_data_1[:dev_size]
    test_data_1 = test_data_1[dev_size:]

    dev_data_2 = test_data_2[:dev_size]
    test_data_2 = test_data_2[dev_size:]
    # train_data_not_binds = train_data_not_binds[:len(train_data_binds)]

    dev_data = np.concatenate((np.asarray(dev_data_2), np.asarray(dev_data_1)))
    np.random.shuffle(dev_data)

    test_data = np.concatenate((np.asarray(test_data_2), np.asarray(test_data_1)))
    np.random.shuffle(test_data)

    np.random.shuffle(train_data_not_binds)
    train_data = np.concatenate((np.asarray(train_data_not_binds[:400000]), np.asarray(train_data_binds)))
    np.random.shuffle(train_data)
    np.savetxt('train_data_binds_142_unbalanced.txt', train_data_binds, fmt='%s')
    np.savetxt('train_data_not_binds_142_unbalanced.txt', train_data_not_binds, fmt='%s')
    # np.savetxt('train_data_binds42_unbalanced.txt', train_data, fmt='%s')
    # np.savetxt('dev_data_142.txt', dev_data, fmt='%s')
    # np.savetxt('test_data_142.txt', test_data, fmt='%s')
    exit()
    return train_data_binds,train_data_not_binds, dev_data, test_data


def get_pre_trained_vectors(vec_file):
    tri_dict = {}
    weights = []
    data = pd.read_csv(vec_file).values
    for row in data:
        row = row[0].strip().split('\t')
        weight = np.array([float(val) for val in row[1:]])
        weights.append(weight)
        tri_dict[row[0].lower()] = weight
    tri_dict['#'] = torch.zeros(100)
    weights.append(torch.zeros(100))
    return tri_dict, torch.FloatTensor(weights)
