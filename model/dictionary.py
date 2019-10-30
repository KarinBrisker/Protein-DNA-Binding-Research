import random
from torch.utils.data import Dataset
import torch


"""
class to keep the data for dna and protein - their amino acids and nucleotides ABC etc.
"""


class Dictionary(object):
    def __init__(self):
        self.dna_vocab = ['a', 'c', 'g', 't']
        self.idx2dna = dict(enumerate(self.dna_vocab))
        self.dna2idx = dict(zip(self.idx2dna.values(), self.idx2dna.keys()))

        # uni-grams
        self.amino_acids = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v',
                            'w', 'y', '#']
        self.idx2amino_acids = dict(enumerate(self.amino_acids))
        self.amino_acids2idx = dict(zip(self.idx2amino_acids.values(), self.idx2amino_acids.keys()))

        self.protein2seq = {}

        lines = open('../DNA_data/protein_seq.txt').readlines()
        for line in lines:
            # line = name <sep> seq
            line = line.split()
            # prot2val:dict. prot name to sequence
            self.protein2seq[line[0]] = line[1].strip().lower()

        self.proteins = list(self.protein2seq.keys())

    def __len__(self):
        return len(self.idx2amino_acids)

    def len_dna(self):
        return len(self.idx2dna)


"""
batch contains proteins, proteins2, dnas, dnas2, labels
"""


def get_dataset_info(batch):
    return batch[:, 1:201].clone().detach().view(-1, 200).long(), \
           batch[:, 201:401].clone().detach().view(-1, 200).long(), \
           batch[:, 401:417].clone().detach().view(-1, 16).long(), \
           batch[:, 417:433].clone().detach().long().view(-1, 16), \
           batch[:, 433].clone().detach().view(-1, 1)


"""
this class provides two random samples of: dna, protein, binding score
output for each sample:
        # protein - protein seq
        # dna - dna seq
        # amino acids - 12 features for each amino acid in the protein
        # label - binding score
"""


class ProteinVectorsDataset(Dataset):
    def __init__(self, proteins, proteins_names, dna, label, amino_acids):
        self.amino_acids = amino_acids
        all_proteins = torch.stack(list(proteins))
        all_dnas = torch.stack(list(dna))
        all_labels = torch.tensor(list(label))
        combined = list(zip(list(proteins), list(proteins_names), list(dna), list(label)))
        random.shuffle(combined)
        proteins2, proteins_names2, dna2, label2 = zip(*combined)
        all_proteins2 = torch.stack(list(proteins2))
        all_dnas2 = torch.stack(list(dna2))
        all_labels2 = torch.tensor(list(label2))
        diff = abs(all_labels - all_labels2) > 0.2
        label = torch.tensor([1 if (all_labels[i] - all_labels2[i]) > 0 else 0 for i in range(len(diff))])
        batch = torch.cat([diff.view(-1,1).float(), all_proteins.float(), all_proteins2.float(), all_dnas.float(),
                           all_dnas2.float(), label.view(-1,1).float()], dim=1)
        batch = batch[(batch[:, 0] == 1).nonzero().squeeze(1)]
        proteins, proteins2, dnas, dnas2, labels = get_dataset_info(batch)
        self.amino_acids = amino_acids
        self.all_proteins = proteins
        self.all_dnas = dnas
        self.all_labels = labels
        self.amino_acids1 = [self.amino_acids[x] for x in proteins_names]
        self.all_proteins2 = proteins2
        self.all_dnas2 = dnas2
        self.amino_acids2 = [self.amino_acids[x] for x in proteins_names2]

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        protein = self.all_proteins[idx]
        dna = self.all_dnas[idx]
        label = self.all_labels[idx]
        amino_acids = self.amino_acids1[idx]

        protein2 = self.all_proteins2[idx]
        dna2 = self.all_dnas2[idx]
        amino_acids2 = self.amino_acids2[idx]

        return protein, dna, label, amino_acids, protein2, dna2, amino_acids2
