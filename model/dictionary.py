import torch
from torch.utils.data import Dataset
import re
import random


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


class ProteinVectorsDataset(Dataset):
    def __init__(self, proteins, dna, label, amino_acids):
        print('a1')
        self.amino_acids = amino_acids
        self.all_proteins = torch.stack(list(proteins))
        print('a2')
        self.all_dnas = torch.stack(list(dna))
        print('a3')
        self.all_labels = torch.tensor(list(label))
        print('a4')
        self.amino_acids1 = list(map(self.process_prot_2_amino_acids_embeddings, proteins))
        print('a5')
        combined = list(zip(list(proteins), list(dna), list(label)))
        print('a6')
        random.shuffle(combined)
        print('a7')

        proteins2, dna2, label2 = zip(*combined)
        print('a8')
        self.all_proteins2 = torch.stack(list(proteins2))
        print('a9')
        self.all_dnas2 = torch.stack(list(dna2))
        print('a0')
        self.all_labels2 = torch.tensor(list(label2))
        print('a1')
        self.amino_acids2 = list(map(self.process_prot_2_amino_acids_embeddings, proteins2))

    def process_prot_2_amino_acids_embeddings(self, protein):
        return self.amino_acids[re.sub(r"[\n\t\s]*", "", str(protein))[7:-1]]

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