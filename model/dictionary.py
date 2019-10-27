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

        # 3 - grams
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
        return len(self.idx2protein)

    def len_dna(self):
        return len(self.idx2dna)


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
        self.all_proteins = torch.stack(list(proteins))
        self.all_dnas = torch.stack(list(dna))
        self.all_labels = torch.tensor(list(label))
        self.amino_acids1 = [self.amino_acids[x] for x in proteins_names]
        print('done first')
        combined = list(zip(list(proteins), list(proteins_names), list(dna), list(label)))
        random.shuffle(combined)
        proteins2, proteins_names2, dna2, label2 = zip(*combined)
        self.all_proteins2 = torch.stack(list(proteins2))
        self.all_dnas2 = torch.stack(list(dna2))
        self.all_labels2 = torch.tensor(list(label2))
        self.amino_acids2 = [self.amino_acids[x] for x in proteins_names2]
        print('done second')

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        protein = self.all_proteins[idx]
        dna = self.all_dnas[idx]
        label = self.all_labels[idx]
        amino_acids = self.amino_acids1[idx]

        protein2 = self.all_proteins2[idx]
        dna2 = self.all_dnas2[idx]
        label2 = self.all_labels2[idx]
        amino_acids2 = self.amino_acids2[idx]

        return protein, dna, label, amino_acids, protein2, dna2, label2, amino_acids2
